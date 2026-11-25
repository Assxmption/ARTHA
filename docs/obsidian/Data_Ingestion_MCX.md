# Data: MCX Commodities Ingestion

**Status**: Architecture & Event Mapping Defined  
**Path**: `app/data/mcx.py` (Planned), `app/data/news.py`  

## Overview
ARTHA's charter explicitly encompasses both **Indian Equities (NSE)** and **Commodities (MCX)**. The Multi Commodity Exchange of India (MCX) provides trading in energy, precious metals, base metals, and agricultural commodities.

Commodities intelligence differs structurally from equities fundamentals: commodities have no financial statements or balance sheets. Instead, commodity price discovery is governed by:
1. **Physical Supply & Demand Shocks**: Global production quotas (OPEC+), export bans, inventory build-ups, and refinery outages.
2. **Macroeconomic Drivers**: Interest rate expectations, USD/INR foreign exchange rates, inflation indices (CPI / WPI), and central bank reserve accumulation.
3. **Weather & Geopolitical Events**: Monsoon progression, agricultural harvest reports, and conflict-driven trade route disruptions.

---

## Core Focus Commodities (MCX Universe)

| Commodity Symbol | Sector | Primary Drivers | Linked Equity Sectors |
| :--- | :--- | :--- | :--- |
| `CRUDEOIL` | Energy | OPEC+ quotas, US SPR inventories, geopolitical shipping routes | OMCs (BPCL, IOC, HPCL), Paints (Asian Paints), Aviation (IndiGo) |
| `NATURALGAS` | Energy | Winter heating demand, European storage inventories, LNG feedgas | City Gas (IGL, MGL), Fertilizers (Chambal, GNFC) |
| `GOLD` | Precious Metals| US 10Y real yields, USD strength, Central bank purchases, RBI reserves | Jewellery (Titan, Kalyan), NBFCs (Muthoot, Manappuram) |
| `SILVER` | Precious Metals| Industrial solar PV demand, electronics manufacturing, macro store-of-value | Electronics, Industrial Component Manufacturers |
| `COPPER` | Base Metals | Global manufacturing PMIs, Chinese infrastructure, EV transition | Cables (Polycab, Havells), Capital Goods (Siemens, ABB) |

---

## Event Tagging & Sentiment Integration
In the current implementation, commodity intelligence is operationalized through the [[Data_News]] and [[Sentiment_Agent]] modules:
- **`EventType.SUPPLY_SHOCK`**: Keyword classification explicitly monitors commodity news terms (`"opec"`, `"crude"`, `"inventory"`, `"monsoon"`, `"harvest"`, `"production cut"`, `"demand"`, `"inflation"`, `"cpi"`, `"wpi"`).
- **FinBERT Local Scoring**: Transferred to commodity market headlines to score macro supply/demand pressures without LLM token waste.
- **Cross-Market Factor Spreads**: Commodity price spikes are fed as input signals to equity factor models (e.g., rising crude as a negative input for paint/aviation margins).

---

## System Connections
- **Data Layer Hierarchy**: Sits under [[Data_Layer]].
- **Sentiment & Event Flow**: Consumed by [[Sentiment_Agent]] via `EventType.SUPPLY_SHOCK`.
- **Macro Risk**: Cross-referenced by [[Risk_Agent]] when evaluating sector-specific exposure constraints.
