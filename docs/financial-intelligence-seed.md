# QID Financial Intelligence - Market Regime Analysis

## Overview

The QID Financial Intelligence Engine monitors macro market conditions across multiple asset classes and classifies market regimes in real-time. The system uses Crucix, an automated macro sweep engine, to collect data every 15 minutes during market hours.

## Market Regime Classification

The system classifies markets into the following regimes:

- **Normal**: VIX below 20, balanced cross-asset dynamics
- **Elevated**: VIX between 20-30, increased volatility and correlation breakdowns
- **Energy Shock**: WTI crude oil moves exceeding 2% intraday, combined with elevated VIX
- **Danger Zone**: VIX above 25, extreme volatility requiring trading suppression

## Key Indicators

### Volatility
- **VIX (CBOE Volatility Index)**: Primary fear gauge. Above 25 enters the "danger zone"
- **VIX Term Structure**: Contango vs backwardation signals forward expectations

### Equities
- **S&P 500**: Benchmark equity index. Large daily moves (>2%) signal regime transitions
- **Market Breadth**: Participation across sectors indicates rally durability

### Fixed Income
- **TLT (20+ Year Treasury Bond ETF)**: Long duration bond proxy
- **Yield Curve**: Inversion signals recession risk
- **HYG (High Yield Corporate Bond ETF)**: Credit risk appetite indicator

### Commodities
- **WTI Crude Oil**: Energy price shocks propagate across all asset classes
- **OPEC Dynamics**: Supply decisions affect WTI directly
- **Refinery Margins**: Physical market fundamentals

### Geopolitical Risk
- **Conflict Events**: Armed conflicts affect energy security and risk appetite
- **Sanctions**: Trade restrictions impact commodity flows
- **Central Bank Policy**: Fed decisions transmit through yield curves to risk assets

## Market Participants

The financial intelligence engine models 8 market participant archetypes:

1. **Marcus Sterling (Global Macro Fund Manager)**: 20-year veteran running $2B global macro. Focuses on cross-asset correlations, central bank policy, and geopolitical risk. Known for contrarian calls during regime transitions.

2. **Sarah Chen (Systematic CTA / Quant Strategist)**: PhD in financial mathematics. Runs trend-following and mean-reversion strategies across futures. Obsessed with volatility regimes and statistical anomalies.

3. **Jake Morrison (Market Maker / Prop Trader)**: Former CME floor trader. Reads orderflow, delta, and gamma exposure. Pure price action and positioning focus.

4. **Dr. Elena Volkov (Geopolitical Risk Analyst)**: Former intelligence analyst. Specializes in energy security, conflict escalation, and sanctions impacts. Connects world events to market consequences.

5. **Tom Retail (Retail Trader / Influencer)**: Self-taught with 50K followers. Trades momentum and breakouts. Represents retail crowd sentiment — excited during rallies, panicked during selloffs.

6. **Catherine Wells (Fixed Income Strategist / Fed Watcher)**: 15 years covering rates and central bank policy. Parses every Fed statement, dot plot, and minutes release.

7. **Raj Patel (Energy Sector Analyst)**: Former petroleum engineer. Deep expertise in oil markets, OPEC dynamics, refinery capacity, and energy transition.

8. **Alexis Dubois (Chief Risk Officer)**: Runs risk management for multi-strategy hedge fund. Survived 2008, 2020, and every vol spike in between. Focuses on tail risk and correlation regimes.

## Trading Signals

The regime classification produces directional signals:
- **Bias Direction**: long, short, or neutral based on regime conditions
- **ML Score Threshold**: Adjusted dynamically (0.55-0.70) based on regime
- **Trading Suppression**: Activated during extreme conditions (danger zone)

## Recent Market Conditions (April 2026)

The market is currently in an elevated/energy_shock regime:
- VIX at 25.8 (danger zone boundary)
- WTI crude oil volatile, ranging $95-116 with large intraday swings
- S&P 500 showing sharp rallies (+4.3%) interpreted as short-covering squeezes
- TLT bonds declining, suggesting rate pressure
- HYG high-yield credit rising, showing some risk appetite
- Directional bias: short due to elevated volatility
