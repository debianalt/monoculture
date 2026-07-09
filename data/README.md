# Data dictionary

## geocoded_sociedades.parquet

Main organisational registry. 14,177 records geocoded from the Misiones subset (by registered fiscal domicile) of the Registro Nacional de Sociedades, Argentina.

| Column | Type | Description |
|--------|------|-------------|
| `cuit` | str | Tax identifier (CUIT) |
| `razon_social` | str | Registered name |
| `tipo_societario` | str | Juridical form (raw) |
| `fecha_hora_contrato_social` | datetime | Date of incorporation |
| `actividad_estado` | str | Fiscal status: AC (active), BD (baja definitiva = cancelled) |
| `actividad_codigo` | str | CLAE6 economic activity code |
| `actividad_descripcion` | str | Economic activity description |
| `dom_fiscal_localidad` | str | Fiscal domicile locality |
| `dom_fiscal_calle` | str | Fiscal domicile street |
| `dom_fiscal_numero` | str | Fiscal domicile number |
| `dom_fiscal_cp` | str | Postal code |
| `departamento` | str | Department (17 departments in Misiones) |
| `lat` | float | Latitude (WGS84) |
| `lon` | float | Longitude (WGS84) |
| `osm_type` | str | Geocoding precision (street, locality, department) |

**Source**: Registro Nacional de Sociedades, published by the Ministerio de Justicia on the Argentine national open-data portal (datos.gob.ar) and compiled from ARCA fiscal records; file `registro-nacional-sociedades-20260223.csv`, retrieved 23 February 2026. Filtered to `dom_fiscal_provincia == MISIONES` (14,278 unique CUITs of 3,050,044 national activity records). Period: 1901-2025.

## enriquecido_arca_v2.parquet

Fiscal enrichment from ARCA (formerly AFIP), the Argentine federal tax authority. 6,652 organisations with active CUIT resolved.

| Column | Type | Description |
|--------|------|-------------|
| `cuit` | str | Tax identifier |
| `empleador` | bool | Employer status |
| `iva_categoria` | str | IVA (VAT) registration category |
| `actividad_principal` | str | Primary declared economic activity |

**Source**: ARCA/AFIP fiscal system. Queried 2025 — that is, *before* the
registry snapshot of 23 February 2026 was retrieved, not after it. No claim in
the paper rests on this file (see the note under Tables in the top-level README).

## arg_gdp_growth.csv

Argentina, annual growth of real gross domestic product, 1961–2025.

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Calendar year |
| `gdp_growth_pct` | float | Annual growth of real GDP, per cent |

**Source**: World Bank, *World Development Indicators*, series
NY.GDP.MKTP.KD.ZG, retrieved 9 July 2026. Exogenous to the registry and to the
normative acts of Table 1; used only as the covariate against which the
counter-cyclical account of cooperative entry is weighed. Aggregated by
`code/gdp_by_period.py`.

## lookup_localidad_departamento.csv

Mapping of 283 localities to 17 departments in Misiones.

| Column | Type | Description |
|--------|------|-------------|
| `localidad` | str | Locality name |
| `departamento` | str | Department name |

**Source**: INDEC (Argentine national statistics institute).
