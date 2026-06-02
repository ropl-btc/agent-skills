# Risk And Liquidation

Use this when answering borrow-capacity, LTV, health, repayment, add-collateral, or liquidation-risk questions.

## Core Concepts

- Liquidium loans are over-collateralized.
- LTV is loan-to-value: borrowed value divided by collateral value.
- Max LTV is the highest allowed starting/validation LTV for the selected collateral/borrow pair.
- Liquidation threshold is the risk threshold where collateral can be sold to repay debt.
- Liquidation price is the estimated collateral asset price where the current debt/collateral pair reaches the liquidation threshold.
- Health factor/portfolio health shows how far a profile position is from liquidation.
- Borrow APY/interest rate is dynamic and can change with pool utilization.
- Interest accrues continuously and compounds; debt grows over time.
- There is no fixed repayment date, but liquidation risk can rise as prices move or debt grows.

## Agent Warning Policy

Always mention risk when:

- the user asks how much they can borrow
- the user asks for a high LTV
- the quote is close to max LTV
- the user asks to proceed with a borrow
- the user asks about repayment or adding collateral

Suggested concise warning:

> Liquidation risk: if your LTV reaches the liquidation threshold, collateral can be sold to repay the loan.

If LTV is near max:

> This is close to the max LTV, so a price move or accrued interest could put the loan at risk.

If giving instant-loan funding instructions:

> Send only the specified collateral asset on the specified chain before the deposit deadline.

If giving repayment instructions:

> Interest accrues continuously, so refresh the repayment amount before sending.

## Instant Loan Specifics

- Run `quote` before `instant-create`.
- Run `max-borrow` for capacity questions.
- Do not represent `max-borrow` as recommended borrow size; it is a limit estimate from current data.
- For stablecoin borrows such as USDC or USDT against another collateral asset, calculate and show the estimated liquidation price for the collateral asset when quote data is available.
- Requote immediately before creation.
- Explain that if LTV moves too high before the collateral deposit is registered, the loan may not open until more collateral is supplied or may be refunded according to the protocol/app behavior.
- Deposit target opens/adds collateral.
- Repayment target pays debt down.

## Repayment And Add Collateral

Repaying reduces debt and improves LTV. Adding collateral increases collateral value and improves LTV. They use different targets.

For instant loans, partial repayment can reduce LTV, but it does not trigger collateral withdrawal. Full collateral release requires the debt to be fully repaid. Always refresh with `loan-instructions --action repay` before telling the user the full repayment amount.

For instant loans, collateral top-ups can be sent to the collateral deposit target even after the loan has started. This is not repayment; it lowers LTV and improves loan health.

## Capacity Answer Shape

When a user asks "how much can I borrow with X BTC?", answer with:

- collateral amount, asset, and chain
- borrow asset and chain
- estimated maximum borrow amount
- current/max LTV used by the quote
- liquidation threshold LTV
- estimated collateral liquidation price when borrowing USDC or USDT against another collateral asset
- current borrowing interest rate/APY if available
- warning that it is live market data and must be requoted before execution
- suggestion to borrow less than the maximum for liquidation buffer

## Profile Health

For profile flows, use:

```ts
client.positions.getHealthFactor(profileId);
client.positions.getUserPositionSummary(profileId);
client.positions.getUserReserves(profileId);
```

Explain health in user terms: higher health is safer; near liquidation is risky; collateral prices, debt value, and liquidation thresholds determine health.

## Liquidation Price Estimate

For a stablecoin borrow against a non-stable collateral asset:

```text
estimated liquidation collateral price =
  current collateral price * current LTV / liquidation threshold
```

Use the SDK quote's current LTV and the collateral pool's `liquidationThreshold`. Treat the result as an estimate only. It changes as debt accrues, stablecoin prices move, collateral prices move, and protocol/oracle data updates.
