# Instant Loans

Use instant loans when the user wants to borrow without creating a Liquidium profile or connecting a wallet. This is the default Liquidium borrow UX and works well for manual wallets, air-gapped wallets, hardware wallets, exchange accounts, and agent-guided workflows.

## Manual Wallet Flow

Use this when the user controls their own wallet and the agent only guides the process.

1. For "how much can I borrow with X collateral?", run `max-borrow` with live data.
2. Explain collateral asset/chain, borrow asset/chain, estimated maximum borrow, current borrowing interest rate/APY, max LTV, liquidation threshold LTV, estimated collateral liquidation price when available, and that the estimate must be requoted before execution.
3. If the user proposes a borrow amount, run `quote` and check `validationErrors`.
4. If LTV is high, recommend a lower borrow amount or more collateral. Read `risk-and-liquidation.md` for warning copy.
5. When the user wants to proceed, ask for:
   - borrow destination address on the borrow asset chain
   - refund address on the collateral asset chain
6. Before loan creation, restate collateral asset/chain/amount, borrow asset/chain/amount, borrow destination, refund destination, current borrowing interest rate/APY, current LTV, max allowed LTV, liquidation threshold LTV, estimated collateral liquidation price when available, deposit window, estimated absolute deadline when available, and the liquidation warning.
7. After explicit confirmation, run `instant-create`. It saves `~/.local/share/liquidium-borrow/loans/<ref>.json`.
8. Return the exact funding instructions from the CLI/SDK:
   - loan ref and local record path
   - collateral amount in display units and base units
   - collateral asset and chain
   - deposit target
   - borrowed amount in display units and base units
   - borrow asset and chain
   - borrow destination
   - refund destination
   - current LTV, max LTV, liquidation threshold LTV, estimated collateral liquidation price, current borrowing interest rate/APY
   - deposit window, estimated absolute deadline, and status
9. Tell the user to send only the specified collateral asset/chain/amount to the deposit target. The final line of the response should be exactly actionable: `Send <amount> <asset> to <deposit address>`.
10. Do not tell the user to use the deposit target for repayment.
11. Explain that the refund address receives collateral returns after full repayment, or refunds if collateral cannot be applied, including failed/late/expired deposits.

## Wallet-Access Agent Flow

Use this when the agent has access to a wallet tool such as a CLI wallet. The default Liquidium route can still be accountless instant loans; the wallet tool only broadcasts the user's collateral or repayment transaction.

1. Run `max-borrow` or `quote` and explain risk.
2. Ask for or derive borrow/refund addresses. Verify the addresses match the requested asset/chain.
3. Restate the full plan and ask for explicit user confirmation before creating the loan.
4. Run `instant-create`; save the generated `~/.local/share/liquidium-borrow/loans/<ref>.json` record.
5. Before sending collateral, restate exact send amount, asset, chain, deposit target, receiving borrow address, expected borrow amount, current LTV, max LTV, liquidation threshold LTV, estimated collateral liquidation price, current borrowing interest rate/APY, and liquidation warning.
6. After explicit confirmation, use the wallet tool to send only the exact collateral amount to `loan.depositTarget`.
7. Store the txid with `loan-tx --ref <ref> --kind collateral --txid <txid>`.
8. Poll with `deposit-status --ref <ref> --txid <txid>` and `loan-instructions --ref <ref> --action status` until active or until the user asks to stop.
9. For repayment, run `loan-instructions --ref <ref> --action repay`, ask for explicit confirmation, use the wallet tool to send repayment, then store the txid with `loan-tx --ref <ref> --kind repayment --txid <txid>`.

## Commands

From the skill directory:

```bash
<skill-path>/scripts/liquidium-borrow max-borrow --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005
<skill-path>/scripts/liquidium-borrow quote --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005 --borrow-amount-decimal 9
<skill-path>/scripts/liquidium-borrow instant-create --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005 --borrow-amount-decimal 9 --borrow-destination 0x2222222222222222222222222222222222222222 --refund-destination bc1qrefunddestination
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action status
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action repay
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action add-collateral
<skill-path>/scripts/liquidium-borrow deposit-status --ref 8Y9AQQ --txid <txid>
<skill-path>/scripts/liquidium-borrow instant-activities --ref 8Y9AQQ --filter active
<skill-path>/scripts/liquidium-borrow instant-find --address bc1q...
<skill-path>/scripts/liquidium-borrow loan-list
<skill-path>/scripts/liquidium-borrow loan-show --ref 8Y9AQQ
<skill-path>/scripts/liquidium-borrow loan-tx --ref 8Y9AQQ --kind collateral --txid <txid>
```

`instant-create`, `instant-get`, and `loan-instructions` save or update the local loan record unless `--no-save` is passed.

## Repayment

Always refresh immediately before giving repayment instructions:

```bash
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action repay
```

Repayment target is `loan.repayment.target`, not `loan.depositTarget`. The current full repayment amount includes accrued interest and buffers where the SDK provides them. Interest accrues continuously, so stale repayment amounts can be wrong.

Partial repayments reduce debt and may reduce LTV, but they do not trigger collateral withdrawal. The loan must be repaid in full to receive the full collateral amount back at the refund address. If the user asks to fully close a loan, use the latest full repayment amount from the refreshed loan.

## Add Collateral

Use add-collateral instructions when the user wants to lower LTV or improve loan health:

```bash
<skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action add-collateral
```

Adding collateral uses the collateral deposit target. It is not repayment and should not be sent to the repayment target. If the user sends more collateral to the deposit target after the loan has started, it tops up the position, lowers LTV, and improves loan health.

## Status And States

Use `loan-instructions --action status`, `instant-get`, and `instant-activities`.

When the user asks whether Liquidium detected their collateral deposit, prefer:

```bash
<skill-path>/scripts/liquidium-borrow deposit-status --ref 8Y9AQQ --txid <txid>
```

If the user has no txid, omit `--txid`. This command combines `instantLoans.get` with `activities.list({ shortRef, filter: "all" })`, saves the latest status locally, and reports whether a deposit activity is visible, whether the provided txid matched Liquidium activity, current/required confirmations, and the loan status.

Explain detection this way:

- If deposit activity exists, say Liquidium has detected the collateral deposit and report activity status (`pending`, `detected`, `processing`, `confirmed`, or `failed`) plus confirmations when available.
- If activity is still empty but the user has a blockchain txid, say Liquidium has not surfaced it in the SDK yet and that detection can take a few minutes after broadcast or confirmation. If a Bitcoin explorer/tool is available, verify the tx separately against the deposit address and amount.
- If the loan remains `awaiting_deposit` but a deposit activity exists, trust and report the activity feed as the more granular deposit-tracking surface; keep polling until the loan becomes `deposit_detected`, `active`, `closed`, or failed/refunded.

Explain states this way:

- `awaiting_deposit`: the loan exists and waits for collateral. Show deposit target, deposit window, and estimated absolute deadline when available.
- `deposit_detected`: collateral has been seen and the borrow is processing. Keep polling.
- `active`: borrow is open. User can repay or add collateral.
- `settling`: closing or processing. Avoid duplicate actions.
- `closed`: loan is finished. Stop prompting for repayment.

Product docs also mention user-facing states like "deposit too small" and "repaid"; if activities or loan state indicate those, explain what additional collateral or no further action is needed.

If collateral is sent after the deposit window or cannot be applied to the loan, explain that the specified refund address is where collateral refunds/returns are directed. If collateral is sent to the deposit address after the loan has started, explain that it is treated as a collateral top-up, not a refund and not a repayment.

## Local Records

Saved records live in `~/.local/share/liquidium-borrow/loans/<ref>.json` by default and should include request details, latest loan state, transfer instructions, deposit deadline estimate, risk fields, txids, and timestamps. local loan records live outside the repo.

Use `LIQUIDIUM_CLI_DATA_DIR` when the host wants records outside the skill directory.

## Recovery

If the user lost the reference, use address recovery:

```bash
<skill-path>/scripts/liquidium-borrow instant-find --address <borrow-or-refund-address>
```

Candidates are not canonical. Hydrate the selected candidate with `instant-get` or `loan-instructions` before showing targets or repayment amounts.

Liquidium product docs mention transaction-ID recovery, but the current SDK path reviewed exposes address recovery, not a public `findByTxid` helper. Store txids locally with `loan-tx` for future local lookup.
