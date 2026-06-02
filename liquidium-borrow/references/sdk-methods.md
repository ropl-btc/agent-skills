# Liquidium SDK Methods

This reference captures practical method shapes for Liquidium borrow and supply work. Prefer local installed types when available.

## Installation And Client

```bash
npm install @liquidium/client
```

```ts
import { LiquidiumClient, LiquidiumError } from "@liquidium/client";

const client = new LiquidiumClient({});
```

Useful config:

```ts
const client = new LiquidiumClient({
  environment: "mainnet",
  apiBaseUrl: "https://your-service.example.com",
  evmRpcUrl: "https://mainnet.infura.io/v3/<key>",
  timeoutMs: 30_000,
});
```

Use default config for normal mainnet SDK flows. Add `evmRpcUrl` or `evmPublicClient` for ETH contract-interaction supply planning. Add `apiBaseUrl` only for custom service deployments or methods that rely on the Liquidium API service.

## Bundled CLI

Prefer the CLI for quick agent operations:

```bash
<skill-path>/scripts/liquidium-borrow --help
<skill-path>/scripts/liquidium-borrow pools
<skill-path>/scripts/liquidium-borrow prices
<skill-path>/scripts/liquidium-borrow max-borrow --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005
<skill-path>/scripts/liquidium-borrow quote --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005 --borrow-amount-decimal 9
<skill-path>/scripts/liquidium-borrow instant-get --ref 8Y9AQQ
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action status
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action repay
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action add-collateral
<skill-path>/scripts/liquidium-borrow repay-instructions --ref 8Y9AQQ
<skill-path>/scripts/liquidium-borrow add-collateral-instructions --ref 8Y9AQQ
<skill-path>/scripts/liquidium-borrow instant-find --address bc1q...
<skill-path>/scripts/liquidium-borrow loan-list
<skill-path>/scripts/liquidium-borrow loan-show --ref 8Y9AQQ
<skill-path>/scripts/liquidium-borrow loan-tx --ref 8Y9AQQ --kind collateral --txid <txid>
<skill-path>/scripts/liquidium-borrow profile-summary --profile-id <profile-id>
```

`instant-create` creates an accountless loan request and returns transfer targets, but it does not move collateral:

```bash
<skill-path>/scripts/liquidium-borrow instant-create \
  --collateral-asset BTC \
  --borrow-asset USDC \
  --collateral-amount-decimal 0.0005 \
  --borrow-amount-decimal 9 \
  --borrow-destination 0x2222222222222222222222222222222222222222 \
  --refund-destination bc1qrefunddestination
```

Environment/config flags:

- `--api-base-url` or `LIQUIDIUM_API_BASE_URL`
- `--environment`
- `--evm-rpc-url` or `LIQUIDIUM_EVM_RPC_URL`
- `--timeout-ms` or `LIQUIDIUM_TIMEOUT_MS`
- `LIQUIDIUM_CLI_CACHE_DIR` for the auto-installed Node dependency cache
- `LIQUIDIUM_CLI_DATA_DIR` for local loan records

By default, the CLI installs `@liquidium/client` into `~/.cache/liquidium-borrow/node` on first real use. That cache is not part of the skill package and can be deleted safely.

By default, `instant-create`, `instant-get`, and `loan-instructions` save or update local records under `~/.local/share/liquidium-borrow/loans`. That directory is gitignored. Use `--no-save` to disable local record writes.

For operational workflows, read:

- `instant-loans.md` for accountless borrow, repay, add-collateral, wallet-agent, status, and recovery flows.
- `profile-flows.md` for connected-wallet supply, borrow, repay, withdraw, and portfolio flows.
- `risk-and-liquidation.md` for LTV, APY, interest, health, and liquidation warnings.

## Modules

- `client.instantLoans`: accountless instant loans with generated deposit and repayment targets.
- `client.market`: pools, prices, and rates.
- `client.quote`: pure quote and LTV helpers.
- `client.activities`: activity lists and receipt status.
- `client.accounts`: profile lifecycle and wallet/profile lookup.
- `client.lending`: profile-based supply, borrow, withdraw, repay, and inflow reporting.
- `client.positions`: per-pool positions, summaries, health, and aggregate stats.
- `client.history`: user and pool history.

## Instant Loans

Use for accountless/headless borrowing.

```ts
const loan = await client.instantLoans.create({
  collateralPoolId,
  borrowPoolId,
  collateralAsset: "BTC",
  borrowAsset: "USDC",
  collateralAmount,
  borrowAmount,
  ltvMaxBps,
  depositWindowSeconds: 3_600n,
  borrowDestination: { type: "External", address: borrowAddress },
  refundDestination: { type: "External", address: refundAddress },
});
```

Restore and track:

```ts
const loan = await client.instantLoans.get({ ref });
const sameLoan = await client.instantLoans.get({ loanId });
const activities = await client.activities.list({ shortRef: ref, filter: "active" });
const status = await client.activities.getStatus({ shortRef: ref, id });
```

For user reassurance after they send collateral, use the CLI shortcut:

```bash
<skill-path>/scripts/liquidium-borrow deposit-status --ref 8Y9AQQ --txid <txid>
```

It reads current loan state and all instant-loan activities. Deposit inflow activities can expose `status`, `txid`, `confirmations`, and `requiredConfirmations` before the simplified loan status becomes `active`.

Address recovery:

```ts
const candidates = await client.instantLoans.findByAddress(address);
const loan = await client.instantLoans.get({ loanId: candidates[0].loanId });
```

Do not display transfer targets from `findByAddress(...)` candidates. Hydrate first.

## Target Formatting

Transfer targets are discriminated unions. Render by `target.type`.

```ts
function formatSupplyTarget(target: { type: string; address?: string; account?: string }) {
  if (target.type === "nativeAddress") return target.address;
  if (target.type === "icrcAccount") return target.account;
  throw new Error(`Unsupported target type: ${target.type}`);
}
```

Also validate target metadata such as `poolId`, `asset`, `chain`, and `action` when present.

## Amounts

SDK amounts are `bigint` base units.

- BTC: satoshis.
- USDC/USDT: token base units according to the pool's `decimals`.

Use `Pool.decimals` from `client.market.listPools()` for conversion.

```ts
function decimalToBaseUnits(value: string, decimals: number): bigint {
  const [whole, fraction = ""] = value.split(".");
  const padded = fraction.padEnd(decimals, "0").slice(0, decimals);
  return BigInt(whole || "0") * 10n ** BigInt(decimals) + BigInt(padded || "0");
}

function baseUnitsToDecimal(value: bigint, decimals: number): string {
  const scale = 10n ** BigInt(decimals);
  const whole = value / scale;
  const fraction = (value % scale).toString().padStart(decimals, "0");
  return `${whole}.${fraction}`.replace(/\.?0+$/, "");
}
```

## LTV Validation

Fetch market inputs, then calculate LTV before creating a loan or enabling signed borrow.

```ts
const [pools, prices] = await Promise.all([
  client.market.listPools(),
  client.market.getAssetPrices(),
]);

const ltv = client.quote.calculateLtv(
  { collateralPoolId, borrowPoolId, collateralAmount, borrowAmount },
  pools,
  prices
);

if (ltv.validationErrors.length > 0) {
  throw new Error(ltv.validationErrors.map((error) => error.message).join(" "));
}
```

`calculateLtv(...)` is synchronous after market data is fetched.

## Instant Loan Lifecycle

- `awaiting_deposit`: show `loan.depositTarget` and deposit deadline.
- `deposit_detected`: keep polling; borrow is processing.
- `active`: show `loan.repayment.amount` and `loan.repayment.target`.
- `settling`: keep polling and avoid duplicate actions.
- `closed`: show final state and stop prompting for repayment.

Reload before showing repayment instructions so the amount and target are current.

Use `loan-instructions --action repay` instead of cached records when giving repayment instructions. Use `loan-instructions --action add-collateral` when the user wants to lower LTV with more collateral.

For instant loans only:

- `depositTarget` is for initial collateral and later collateral top-ups. Top-ups after the loan starts lower LTV and improve health.
- `repayment.target` is for debt repayment.
- The refund destination receives collateral returned after full repayment, or collateral refunds when funds cannot be applied, including failed/late/expired deposit cases.
- Partial repayments can reduce debt/LTV but do not withdraw collateral; full collateral return requires full repayment.

`quote`, `max-borrow`, `instant-create`, and `loan-instructions` expose risk fields where available:

- `currentLtv` / `currentLtvBps`
- `maxAllowedLtv` / `maxAllowedLtvBps`
- `liquidationThreshold` / `liquidationThresholdBps`
- `liquidationEstimate.estimatedLiquidationCollateralPriceUsd`

For stablecoin borrows such as USDC or USDT against BTC or another non-stable collateral asset, show the estimated liquidation collateral price as an estimate, not a guarantee.

`max-borrow`, `instant-create`, and saved loan records expose current borrow-rate fields where available:

- `expectedBorrowApy` for user display
- `rawBorrowingRate` for machine use only
- `rateDecimals` for formatting raw rates

Display the formatted APY/rate to users and never show raw scaled rate integers as percentages.

`instant-create` and `loan-instructions` expose deposit timing fields where available:

- `depositDeadline.depositWindowSeconds`
- `depositDeadline.estimatedDepositDeadlineAt`
- `depositDeadline.estimated`

The SDK returns the deposit window length, not a canonical absolute deadline in the hydrated loan object. The CLI estimates the absolute deadline from the local creation timestamp plus `depositWindowSeconds`; present it as an estimate unless a canonical event timestamp is available.

## Profile Creation

Use only for connected-wallet profile flows.

```ts
const profileId = await client.accounts.createProfile({
  account: walletAddress,
  chain: "ETH",
  walletAdapter: {
    signMessage: async ({ message }) => wallet.signMessage(message),
  },
});
```

Prepare/sign/submit variant:

```ts
const action = await client.accounts.prepareCreateProfile({ account: walletAddress });
const signature = await wallet.signMessage(action.message);
const profileId = await action.submit({
  signature,
  chain: "ETH",
  account: walletAddress,
});
```

Handle existing profiles by catching the relevant SDK/protocol error and resolving the profile instead of retrying.

## Profile-Based Supply And Repay

Use when the app intentionally manages a Liquidium profile.

```ts
const supplyFlow = await client.lending.supply({
  profileId,
  poolId,
  action: "deposit",
});

if (supplyFlow.type === "transfer") {
  showTarget(supplyFlow.target);
}

await supplyFlow.submit({ txid: "<broadcast-txid>" });
```

Use `action: "repayment"` for repayment. Do not reuse a `deposit` target for repayment.

Automated wallet transfer path:

```ts
const supplyFlow = await client.lending.supply({
  profileId,
  poolId,
  action: "deposit",
  amount,
  account: walletAddress,
  walletAdapter: {
    sendBtcTransaction: async ({ toAddress, amountSats }) =>
      wallet.sendBtcTransaction({ toAddress, amountSats }),
  },
});
```

ETH contract-interaction path:

```ts
const supplyFlow = await client.lending.supply({
  mechanism: "contractInteraction",
  profileId,
  poolId,
  action: "deposit",
  walletAdapter,
  account: evmAddress,
  amount,
});
```

This path needs `evmRpcUrl` or `evmPublicClient`, plus `sendEthTransaction`.

## Profile-Based Borrow

Use only for persistent profile/dashboard integrations.

```ts
const quote = client.quote.getQuote(request, pools, prices);

if (quote.validationErrors.length > 0) {
  throw new Error(quote.validationErrors.map((error) => error.message).join(" "));
}

const outflow = await client.lending.borrow({
  profileId,
  poolId: quote.borrowPoolId,
  amount: quote.borrowAmount,
  receiverAddress,
  signerWalletAddress: walletAddress,
  signerChain: "ETH",
  signerWalletAdapter: {
    signMessage: async ({ message }) => wallet.signMessage(message),
  },
});
```

Display `outflow.id` immediately. `outflow.txid` may be null until broadcast/settlement is available.

## Portfolio

```ts
const positions = await client.positions.listPositions(profileId);
const position = await client.positions.getPosition(profileId, poolId);
const healthFactor = await client.positions.getHealthFactor(profileId);
const stats = await client.positions.getUserStats(profileId);
const summary = await client.positions.getUserPositionSummary(profileId);
const reserves = await client.positions.getUserReserves(profileId);
const maxRepay = await client.positions.getMaxRepayAmount(profileId, poolId, 50n);
```

## Error Handling

```ts
try {
  return await client.instantLoans.get({ ref });
} catch (error) {
  if (error instanceof LiquidiumError) {
    throw new Error(error.message);
  }
  throw error;
}
```

Use exported `LiquidiumErrorCode` values when the app needs separate handling for timeout, transport, validation, or protocol errors.

## Rate Formatting

Rates and risk ratios can be fixed-point values scaled by `rateDecimals`, often `27`. Divide by `10 ** rateDecimals` before percentage formatting.

```ts
function formatScaledRatePercent(
  scaledRate: bigint,
  rateDecimals: bigint,
  fractionDigits = 2
): string {
  const scale = 10n ** rateDecimals;
  const displayScale = 10n ** BigInt(fractionDigits);
  const rounded = (scaledRate * 100n * displayScale + scale / 2n) / scale;
  const whole = rounded / displayScale;
  const fraction = rounded % displayScale;
  return `${whole}.${fraction.toString().padStart(fractionDigits, "0")}%`;
}
```

## Common Debug Checks

- Pool exists and is not frozen.
- Amounts are base units and use the selected pool's decimals.
- `ltv.validationErrors` or `quote.validationErrors` is empty before execution.
- Instant loans use `client.instantLoans`, not `client.lending.borrow`.
- Instant-loan create/get do not require wallet adapters.
- `loan.ref` is persisted before showing transfer instructions.
- Repayment instructions come from the freshly loaded loan, not cached targets.
- Profile supply/repay targets are action-specific.
- ETH contract-interaction supply has `evmRpcUrl` or `evmPublicClient`.
- SDK method names match the installed `@liquidium/client` version.
