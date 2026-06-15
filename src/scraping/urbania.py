"""
Scraper de propiedades de Urbania.pe
=====================================

Este módulo extrae información de propiedades (casas y departamentos) en venta
desde Urbania.pe, incluyendo todas las imágenes del carrusel de cada listing.

Flujo de trabajo:
-----------------
  run_scraper()
    → scrape_listings()          # Recorre páginas de resultados
      → _scrape_page()           # Abre cada página con Selenium
        → _scroll_to_load_images()  # Fuerza carga de imágenes lazy
        → _parse_card()          # Extrae datos de cada tarjeta
          → _extract_card_images()  # Busca todas las URLs de imágenes
    → download_all_images()      # Descarga imágenes a disco
    → save_listings()            # Guarda metadata en JSON

Estructura de datos de un listing:
----------------------------------
  {
      "id": "12345",
      "title": "Departamento en venta en Miraflores",
      "price": 500000,
      "price_text": "S/ 500,000",
      "url": "https://urbania.pe/property/12345",
      "images": [
          "https://img10.naventcdn.com/avisos/.../image1.jpg",
          "https://img10.naventcdn.com/avisos/.../image2.jpg",
      ],
      "bedrooms": 3,
      "bathrooms": 2,
      "area": 120,
      "parking": 1,
      "property_type": "departamento",
      "local_images": ["data/raw/listing_0/image_0.jpg", ...]
  }

Selector CSS clave:
-------------------
  div[data-posting-type='PROPERTY']
  Este atributo distingue propiedades en venta de proyectos de desarrolladores
  (que tienen data-posting-type='DEVELOPMENT').

Problema conocido:
------------------
  Urbania tiene rate limiting. Las páginas 2-4 suelen devolver datos
  cacheados/duplicados. Los datos nuevos aparecen cada 3-4 páginas.
  Se crea un nuevo driver por página para evitar bloqueo por sesión.

Dependencias:
-------------
  - Selenium: renderiza JavaScript y contenido dinámico
  - BeautifulSoup: parsea el HTML renderizado
  - requests: descarga las imágenes del CDN
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UrbaniaScraper:
    """
    Scraping de propiedades (casas y departamentos) de Urbania.pe.

    Uso básico:
        config = {"base_url": "https://urbania.pe", "max_pages": 5}
        scraper = UrbaniaScraper(config, data_dir="data/raw")
        listings = scraper.scrape_listings(max_pages=5)
        scraper.download_all_images()
        scraper.save_listings()

    Atributos:
        config (dict): configuración del scraper
            - base_url (str): URL raíz de Urbania
            - max_pages (int): número máximo de páginas a scrappear
        base_url (str): URL raíz de Urbania (ej: "https://urbania.pe")
        data_dir (Path): carpeta donde se guardan los datos y imágenes
        listings (list[dict]): lista de diccionarios con la info de cada propiedad
        _seen_ids (set[str]): IDs ya vistos para evitar duplicados entre páginas
    """

    # ──────────────────────────────────────────────
    #  CONSTANTES DE FILTRADO
    # ──────────────────────────────────────────────

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    """Extensiones de imagen válidas para aceptar una URL."""

    _BLOCKED_PATTERNS = ("svg", "logo", "icon", "sprite", "empresas", "favicon", "pixel")
    """Patrones en URL que se deben excluir (iconos, logos, etc.)."""

    def __init__(self, config: dict, data_dir: str | Path = "data/raw"):
        """
        Inicializa el scraper.

        Args:
            config: diccionario con configuración. Ejemplo:
                {"base_url": "https://urbania.pe", "max_pages": 10}
            data_dir: carpeta donde se guardarán los datos e imágenes.
                Se crea automáticamente si no existe.
        """
        self.config = config
        self.base_url = config["base_url"]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.listings: list[dict] = []
        self._seen_ids: set[str] = set()

    # ══════════════════════════════════════════════
    #  SELENIUM: Creación del navegador headless
    # ══════════════════════════════════════════════

    def _create_driver(self) -> webdriver.Chrome:
        """
        Crea una instancia nueva de Chrome en modo headless.

        ¿Por qué uno nuevo por página?
        Urbania bloquea sesiones que hacen muchas peticiones seguidas.
        Al crear uno nuevo, se obtiene una nueva sesión y se evita el rate limiting.

        Configuración:
          - headless=new: el navegador no abre ventana visual
          - user-agent: simula un navegador real para evitar bloqueos
          - disable-blink-features=AutomationControlled: oculta que es un bot
          - window-size=1920,1080: tamaño de pantalla para renders completos
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        return webdriver.Chrome(options=chrome_options)

    # ══════════════════════════════════════════════
    #  SCRAPING PRINCIPAL: Recorre las páginas
    # ══════════════════════════════════════════════

    def scrape_listings(self, max_pages: int | None = None) -> list[dict]:
        """
        Recorre múltiples páginas de resultados y extrae las propiedades.

        Flujo por cada página:
          1. Abrir navegador → cargar URL con ?page=N
          2. Esperar a que aparezcan las tarjetas de propiedades
          3. Hacer scroll y clickear carousel para cargar imágenes
          4. Parsear el HTML con BeautifulSoup
          5. Extraer datos de cada tarjeta (_parse_card)
          6. Guardar solo listings nuevos (sin duplicar IDs)

        Args:
            max_pages: número máximo de páginas a scrappear.
                Si no se especifica, usa config["max_pages"].

        Returns:
            Lista de diccionarios con información de cada propiedad.
        """
        max_pages = max_pages or self.config.get("max_pages", 10)
        logger.info(f"Scraping {max_pages} pages...")

        for page in range(1, max_pages + 1):
            prev_count = len(self.listings)
            self._scrape_page(page)
            new_count = len(self.listings)

            # Si no hubo datos nuevos, probablemente estamos en rate limiting
            if new_count == prev_count:
                logger.info("  No new data, waiting 10s...")
                time.sleep(10)
            else:
                time.sleep(8)

        return self.listings

    def _scrape_page(self, page: int):
        """
        Extrae las propiedades de una sola página de Urbania.

        Estructura de la página:
          - Cada propiedad es un <div data-posting-type='PROPERTY'>
          - Ignoramos las tarjetas con data-posting-type='DEVELOPMENT' (proyectos nuevos)
          - Cada tarjeta contiene: precio, características, fotos, y un link al detalle
          - Las fotos están en un carousel flickity con clase flickity-lazyloaded

        Flujo:
          1. Crear driver nuevo (evitar rate limiting)
          2. Navegar a la URL con ?page=N
          3. Esperar hasta 15s a que carguen las tarjetas PROPERTY
          4. Hacer scroll y clickear carousel para cargar todas las imágenes
          5. Buscar todas las tarjetas con BeautifulSoup
          6. Parsear cada tarjeta individualmente
          7. Guardar solo las que no estén en _seen_ids

        Args:
            page: número de página a scrappear (1-indexed).
        """
        driver = self._create_driver()
        try:
            url = f"{self.base_url}/buscar/venta-de-casas-o-departamentos?page={page}"
            logger.info(f"Scraping page {page}: {url}")

            driver.get(url)
            time.sleep(5)

            # Esperar a que aparezcan las tarjetas de propiedades
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-posting-type='PROPERTY']"))
                )
            except Exception:
                logger.warning(f"No PROPERTY cards found on page {page}")
                return

            # Forzar carga de imágenes lazy en el carousel
            self._scroll_to_load_images(driver)

            # Parsear el HTML renderizado por Selenium
            soup = BeautifulSoup(driver.page_source, "html.parser")
            cards = soup.select("div[data-posting-type='PROPERTY']")

            page_count = 0
            for card in cards:
                listing = self._parse_card(card)
                if listing and listing.get("id") not in self._seen_ids:
                    self._seen_ids.add(listing["id"])
                    self.listings.append(listing)
                    page_count += 1

            logger.info(f"  Found {page_count} new listings on page {page}")

        except Exception as e:
            logger.error(f"Error scraping page {page}: {e}")
        finally:
            driver.quit()

    def _scroll_to_load_images(self, driver):
        """
        Fuerza la carga de imágenes lazy en el carousel flickity.

        El carousel de Urbania usa lazy loading, lo que significa que solo
        carga las imágenes visibles. Este método:
          1. Elimina el atributo loading="lazy" de todas las imágenes
          2. Hace click en el botón "Next" del carousel 15 veces
          3. Hace scroll por toda la página para cargar contenido restante

        Args:
            driver: instancia de Selenium WebDriver.
        """
        # Eliminar loading="lazy" de todas las imágenes
        driver.execute_script("""
            document.querySelectorAll('img[loading="lazy"]').forEach(img => {
                img.removeAttribute('loading');
            });
        """)

        # Click en el botón "Next" del carousel flickity para cargar todas las imágenes
        driver.execute_script("""
            document.querySelectorAll('.flickity-slider, .carousel, [class*="gallery"]').forEach(carousel => {
                let nextBtn = carousel.parentElement?.querySelector(
                    '.flickity-button.next, [aria-label="Next"], button[class*="next"]'
                );
                if (nextBtn) {
                    for (let i = 0; i < 15; i++) {
                        nextBtn.click();
                    }
                }
            });
        """)
        time.sleep(2)

        # Scroll por toda la página para cargar contenido restante
        last_height = driver.execute_script("return document.body.scrollHeight")
        current_position = 0
        scroll_step = 500

        while current_position < last_height:
            current_position += scroll_step
            driver.execute_script(f"window.scrollTo(0, {current_position});")
            time.sleep(0.2)
            last_height = driver.execute_script("return document.body.scrollHeight")

        # Volver al tope
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

    # ══════════════════════════════════════════════
    #  PARSEO DE TARJETA: Extraer datos de cada card
    # ══════════════════════════════════════════════

    def _parse_card(self, card) -> dict | None:
        """
        Extrae la información de una tarjeta de propiedad (card).

        Estructura HTML de cada tarjeta:
          <div data-posting-type="PROPERTY" data-id="12345" data-to-posting="/property/...">
            <div class="flickity-slider">
              <img src="https://img10.naventcdn.com/...jpg" class="flickity-lazyloaded">
              <img src="https://img10.naventcdn.com/...jpg" class="flickity-lazyloaded">
            </div>
            <div data-qa="POSTING_CARD_PRICE">S/ 500,000</div>
            <div data-qa="POSTING_CARD_FEATURES">3 dorm. · 2 baños · 120 m²</div>
          </div>

        Selectores CSS utilizados:
          - [data-posting-type='PROPERTY']: identifica tarjetas de propiedades
          - [data-qa='POSTING_CARD_PRICE']: elemento del precio
          - [data-qa='POSTING_CARD_FEATURES']: características
          - img[alt]: imagen principal (el alt contiene el título)

        Filtros de calidad:
          - Se descarta si no tiene precio
          - Se descarta si el precio es "Consultar"
          - Se descarta si no tiene área (area == 0)
          - Se descarta si es terreno u oficina

        Args:
            card: elemento BeautifulSoup de la tarjeta.

        Returns:
            Diccionario con datos de la propiedad, o None si se descarta.
        """
        try:
            # ── ID de la propiedad ──
            card_id = card.get("data-id", "")
            if not card_id:
                return None

            # ── Precio ──
            price_elem = card.select_one("[data-qa='POSTING_CARD_PRICE']")
            if not price_elem:
                return None

            price_text = price_elem.text.strip()
            price = self._parse_price(price_text)
            if price is None or price == 0:
                return None

            # ── Título (viene del alt de la imagen principal) ──
            title = ""
            title_elem = card.select_one("img[alt]")
            if title_elem:
                title = title_elem.get("alt", "")

            # ── Construir diccionario base ──
            listing = {
                "id": card_id,
                "title": title,
                "price": price,
                "price_text": price_text,
                "url": card.get("data-to-posting", ""),
                "images": self._extract_card_images(card),
                "bedrooms": 0,
                "bathrooms": 0,
                "area": 0,
                "parking": 0,
                "property_type": "",
            }

            # ── URL completa (data-to-posting viene como ruta relativa) ──
            if listing["url"] and not listing["url"].startswith("http"):
                listing["url"] = self.base_url + listing["url"]

            # ── Características (dormitorios, baños, m², cochera) ──
            features_elem = card.select_one("[data-qa='POSTING_CARD_FEATURES']")
            if features_elem:
                self._parse_features(features_elem.text.strip(), listing)

            # ── Tipo de propiedad (inferido del título) ──
            img_alt = title.lower()
            if "casa" in img_alt:
                listing["property_type"] = "casa"
            elif "departamento" in img_alt:
                listing["property_type"] = "departamento"
            elif "terreno" in img_alt:
                listing["property_type"] = "terreno"
            elif "oficina" in img_alt or "local" in img_alt:
                listing["property_type"] = "oficina"
            else:
                listing["property_type"] = "otro"

            # ── Filtros: solo casas y departamentos con área ──
            if listing["property_type"] in ["terreno", "oficina"]:
                return None
            if listing["area"] == 0:
                return None

            return listing

        except Exception as e:
            logger.error(f"Error parsing card: {e}")
            return None

    def _extract_card_images(self, card) -> list[str]:
        """
        Extrae URLs de imágenes del carrusel flickity dentro de la card.

        El carousel de Urbania usa flickity y carga imágenes de forma lazy.
        Buscamos URLs en múltiples ubicaciones para no perder ninguna:

        1. img[src] - imágenes ya cargadas
        2. img[data-src] - imágenes lazy-loaded
        3. img[data-flickity-lazyload] - atributo específico de flickity
        4. srcset - imágenes responsive
        5. background-image en CSS inline
        6. Atributos data-* que contengan URLs de naventcdn

        Todas las URLs deben:
          - Contener "naventcdn" (CDN de Navent/Urbania)
          - Tener extensión válida (.jpg, .jpeg, .png, .webp)
          - No contener patrones bloqueados (svg, logo, icon, etc.)

        Args:
            card: elemento BeautifulSoup de la tarjeta.

        Returns:
            Lista de URLs de imágenes limpias (sin query strings).
        """
        seen: set[str] = set()
        images: list[str] = []

        # 1. Buscar en tags img (src, data-src, data-original, data-flickity-lazyload)
        for img in card.find_all("img"):
            for attr in ["src", "data-src", "data-original", "data-flickity-lazyload"]:
                src = img.get(attr, "")
                if not src or "naventcdn" not in src:
                    continue

                src_lower = src.lower()
                if any(pat in src_lower for pat in self._BLOCKED_PATTERNS):
                    continue

                ext = Path(src.split("?")[0]).suffix.lower()
                if ext not in self._IMAGE_EXTENSIONS:
                    continue

                clean = src.split("?")[0]
                if clean not in seen:
                    seen.add(clean)
                    images.append(clean)

        # 2. Buscar en srcset de img y source
        for elem in card.find_all(["img", "source"]):
            srcset = elem.get("srcset", "")
            for part in srcset.split(","):
                url = part.strip().split(" ")[0]
                if not url or "naventcdn" not in url:
                    continue

                url_lower = url.lower()
                if any(pat in url_lower for pat in self._BLOCKED_PATTERNS):
                    continue

                ext = Path(url.split("?")[0]).suffix.lower()
                if ext not in self._IMAGE_EXTENSIONS:
                    continue

                clean = url.split("?")[0]
                if clean not in seen:
                    seen.add(clean)
                    images.append(clean)

        # 3. Buscar en background-image CSS inline
        for elem in card.find_all(style=True):
            style = elem["style"]
            urls_found = re.findall(r'url\(["\']?(https?://[^"\'\)]+?)["\']?\)', style)
            for url in urls_found:
                if "naventcdn" not in url:
                    continue
                url_lower = url.lower()
                if any(pat in url_lower for pat in self._BLOCKED_PATTERNS):
                    continue
                ext = Path(url.split("?")[0]).suffix.lower()
                if ext not in self._IMAGE_EXTENSIONS:
                    continue
                clean = url.split("?")[0]
                if clean not in seen:
                    seen.add(clean)
                    images.append(clean)

        # 4. Buscar en atributos data-* que contengan URLs de imágenes
        for elem in card.find_all(True):
            for attr_name, attr_value in elem.attrs.items():
                if not isinstance(attr_value, str):
                    continue
                if "naventcdn" not in attr_value:
                    continue
                urls_found = re.findall(
                    r'https?://[^"\'\s]+?naventcdn\.com[^"\'\s]+?\.(?:jpg|jpeg|png|webp)',
                    attr_value
                )
                for url in urls_found:
                    url_lower = url.lower()
                    if any(pat in url_lower for pat in self._BLOCKED_PATTERNS):
                        continue
                    clean = url.split("?")[0]
                    if clean not in seen:
                        seen.add(clean)
                        images.append(clean)

        return images

    # ══════════════════════════════════════════════
    #  PARSEO DE CARACTERÍSTICAS
    # ══════════════════════════════════════════════

    def _parse_features(self, text: str, listing: dict):
        """
        Extrae las características numéricas del texto de la tarjeta.

        Formato típico: "3 dorm. · 2 baños · 120 m² · 1 cochera"

        Se usan expresiones regulares para extraer cada valor:
          - Area: número seguido de "m²" o "m2"
          - Dormitorios: número seguido de "dormitorio", "dorm", "hab", "habitacion"
          - Baños: número seguido de "baño"
          - Cochera: número seguido de "cochera", "estacionamiento", "estac"

        Args:
            text: texto de características de la tarjeta.
            listing: diccionario del listing a actualizar con los valores.
        """
        text = text.lower()

        area_match = re.search(r"(\d+)\s*m[²2]", text)
        if area_match:
            listing["area"] = int(area_match.group(1))

        hab_match = re.search(r"(\d+)\s*(?:dormitorio|dorm|hab|habitacion)", text)
        if hab_match:
            listing["bedrooms"] = int(hab_match.group(1))

        banio_match = re.search(r"(\d+)\s*baño", text)
        if banio_match:
            listing["bathrooms"] = int(banio_match.group(1))

        cochera_match = re.search(r"(\d+)\s*(?:cochera|estacionamiento|estac)", text)
        if cochera_match:
            listing["parking"] = int(cochera_match.group(1))

    def _parse_price(self, price_text: str) -> int | None:
        """
        Convierte el texto del precio a un número entero.

        Formatos soportados:
          - "S/ 500,000" → 500000 (Soles peruanos)
          - "S/1,200,000" → 1200000
          - "USD 200,000" → 740000 (convertido a soles con factor 3.7)
          - "Consultar" → None (no se puede determinar el precio)

        Args:
            price_text: texto del precio desde la tarjeta.

        Returns:
            Precio como entero, o None si no se puede determinar.
        """
        try:
            if "consultar" in price_text.lower():
                return None

            # Precio en Soles: "S/ 500,000"
            pen_match = re.search(r"S/\s*([\d,\.]+)", price_text)
            if pen_match:
                cleaned = pen_match.group(1).replace(",", "").replace(".", "")
                return int(cleaned)

            # Precio en dólares: "USD 200,000" → se convierte a soles
            usd_match = re.search(r"USD\s*([\d,\.]+)", price_text)
            if usd_match:
                cleaned = usd_match.group(1).replace(",", "").replace(".", "")
                return int(cleaned) * 3.7

            # Fallback: eliminar todo lo que no sea número
            cleaned = re.sub(r"[^\d]", "", price_text)
            return int(cleaned) if cleaned else None
        except Exception:
            return None

    # ══════════════════════════════════════════════
    #  DESCARGA DE IMÁGENES
    # ══════════════════════════════════════════════

    def download_listing_images(self, listing: dict, idx: int) -> list[str]:
        """
        Descarga las imágenes de una propiedad y las guarda en disco.

        Estructura de carpetas:
          data/raw/
            listing_0/
              image_0.jpg    ← imagen principal
              image_1.jpg    ← imagen 2
              image_2.jpg    ← imagen 3
            listing_1/
              ...

        Se descargan máximo 10 imágenes por propiedad.

        Args:
            listing: diccionario del listing con URLs en "images".
            idx: índice del listing (para nombrar la carpeta).

        Returns:
            Lista de rutas locales de las imágenes descargadas.
        """
        listing_dir = self.data_dir / f"listing_{idx}"
        listing_dir.mkdir(exist_ok=True)

        downloaded: list[str] = []
        for i, img_url in enumerate(listing.get("images", [])[:10]):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(img_url, timeout=10, headers=headers)
                if response.status_code == 200:
                    parsed = urlparse(img_url)
                    ext = Path(parsed.path).suffix or ".jpg"
                    filepath = listing_dir / f"image_{i}{ext}"
                    filepath.write_bytes(response.content)
                    downloaded.append(str(filepath))
            except Exception as e:
                logger.error(f"Error downloading image: {e}")

        return downloaded

    def download_all_images(self) -> list[dict]:
        """
        Descarga imágenes de todos los listings scrapeados.

        Actualiza cada listing con la key "local_images" que contiene
        las rutas locales de las imágenes descargadas.

        Returns:
            Lista de listings actualizados con rutas locales.
        """
        logger.info(f"Downloading images for {len(self.listings)} listings...")
        for idx, listing in enumerate(tqdm(self.listings, desc="Downloading images")):
            listing["local_images"] = self.download_listing_images(listing, idx)
        return self.listings

    # ══════════════════════════════════════════════
    #  GUARDAR / CARGAR LISTINGS
    # ══════════════════════════════════════════════

    def save_listings(self, filename: str = "listings.json"):
        """
        Guarda la lista de listings en un archivo JSON.

        Args:
            filename: nombre del archivo JSON (default: "listings.json").
        """
        import json
        output_file = self.data_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.listings, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.listings)} listings to {output_file}")

    def load_listings(self, filename: str = "listings.json") -> list[dict]:
        """
        Carga listings previamente guardados desde un archivo JSON.

        Args:
            filename: nombre del archivo JSON a cargar.

        Returns:
            Lista de listings cargados.
        """
        import json
        listings_file = self.data_dir / filename
        if listings_file.exists():
            with open(listings_file, "r", encoding="utf-8") as f:
                self.listings = json.load(f)
        return self.listings


# ══════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL: Orquesta todo el proceso
# ══════════════════════════════════════════════

def run_scraper(
    config: dict,
    max_pages: int | None = None,
    data_dir: str | Path = "data/raw",
) -> list[dict]:
    """
    Función principal que ejecuta todo el pipeline de scraping.

    Pasos:
      1. Crear scraper con la configuración
      2. Scrappear listings de Urbania (HTML → diccionarios)
      3. Descargar imágenes de cada listing
      4. Guardar listings.json con las rutas locales

    Uso:
        from src.scraping.urbania import run_scraper

        config = {"base_url": "https://urbania.pe", "max_pages": 5}
        listings = run_scraper(config, data_dir="data/raw/urbania")

    Args:
        config: diccionario con configuración del scraper.
        max_pages: número máximo de páginas (sobreescribe config).
        data_dir: carpeta donde guardar datos e imágenes.

    Returns:
        Lista de listings con información completa e imágenes descargadas.
    """
    scraper = UrbaniaScraper(config, data_dir=data_dir)
    listings = scraper.scrape_listings(max_pages)
    scraper.download_all_images()
    scraper.save_listings()
    return listings
