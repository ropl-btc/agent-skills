# Profile Flows

Use profile flows when the user wants a connected-wallet Liquidium dashboard: supply, borrow, repay, withdraw, repeated borrowing, positions, linked wallets, or profile-level portfolio health. Do not add profile creation to the default accountless instant-loan flow.

## When To Use

- The user wants to manage positions across sessions from one profile.
- The user wants to supply liquidity and earn yield.
- The user wants to borrow repeatedly against existing supplied collateral.
- The user wants to repay or withdraw from a profile position.
- The user wants portfolio summary, health factor, reserves, history, or linked wallets.
- The user explicitly wants an ETH contract-interaction supply flow.

## Portfolio Reads

```bash
<skill-path>/scripts/liquidium-borrow profile-summary --profile-id <profile-id>
<skill-path>/scripts/liquidium-borrow positions --profile-id <profile-id>
```

SDK calls:

```ts
const positions = await client.positions.listPositions(profileId);
const summary = await client.positions.getUserPositionSummary(profileId);
const reserves = await client.positions.getUserReserves(profileId);
const healthFactor = await client.positions.getHealthFactor(profileId);
```

For profile questions, explain that health can change as prices move and interest accrues.

## Profile Creation

Use only when wallet signing is available and the user confirms profile creation.

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

Handle existing profiles by catching the relevant SDK/protocol error and resolving the existing profile instead of retrying.

## Supply And Repay By Deposit Address

Use this when the app/profile should show a target and let the user send funds externally.

```ts
const supplyFlow = await client.lending.supply({
  profileId,
  poolId,
  action: "deposit",
});

showTarget(supplyFlow.target);
await supplyFlow.submit({ txid });
```

Use `action: "repayment"` for repayment. Deposit and repayment targets are action-specific; do not reuse a deposit target for repayment.

## Wallet-Executed Supply

For BTC or transfer-path ETH stablecoin supply, provide only the wallet adapter methods needed.

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

## ETH Contract-Interaction Supply

Use only for account-based ETH stablecoin flows where an EVM wallet should approve and deposit through the SDK.

Requirements:

- `evmRpcUrl` or `evmPublicClient`
- wallet adapter with `sendEthTransaction`
- token amount in base units
- explicit user confirmation before approval/deposit transactions

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

Instant loans return their own deposit and repayment targets; do not use this path for accountless instant loans.

## Profile-Based Borrow

Use quote-first borrowing for connected profiles:

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

Display `outflow.id` immediately. `outflow.txid` may be missing until the protocol assigns or broadcasts the chain transaction. Use activities/history or app-level polling if a txid is needed.

## Withdraw

Use signed profile withdraw flows when the user wants to remove supplied assets. Confirm destination, asset, amount, chain, and risk impact before signing or submitting.

## Safety

- Confirm before every signature or transaction.
- Never ask for private keys or seed phrases.
- For supply, repay, borrow, or withdraw, show asset, chain, amount, destination/target, expected APY/rate where available, and health/risk impact.
- For profile repayment, use the repayment target for `action: "repayment"`; do not reuse deposit target.
