# Data dictionary

## geocoded_sociedades.parquet

Main organisational registry. 14,177 records geocoded from the Inspeccion General de Justicia (IGJ) registry of Misiones, Argentina.

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

**Source**: Inspeccion General de Justicia (IGJ), Argentine national company registry. Period: 1901-2025.

## enriquecido_arca_v2.parquet

Fiscal enrichment from ARCA (formerly AFIP), the Argentine federal tax authority. 6,652 organisations with active CUIT resolved.

| Column | Type | Description |
|--------|------|-------------|
| `cuit` | str | Tax identifier |
| `empleador` | bool | Employer status |
| `iva_categoria` | str | IVA (VAT) registration category |
| `actividad_principal` | str | Primary declared economic activity |

**Source**: ARCA/AFIP fiscal system. Queried 2025.

## lookup_localidad_departamento.csv

Mapping of 283 localities to 17 departments in Misiones.

| Column | Type | Description |
|--------|------|-------------|
| `localidad` | str | Locality name |
| `departamento` | str | Department name |

**Source**: INDEC (Argentine national statistics institute).
