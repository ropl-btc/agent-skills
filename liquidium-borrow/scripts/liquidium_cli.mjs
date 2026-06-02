#!/usr/bin/env node
import { createRequire } from "node:module";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const skillDir = dirname(__dirname);
const homeDir = process.env.HOME || homedir();
if (!homeDir) {
  throw new Error("Cannot determine home directory. Set HOME, LIQUIDIUM_CLI_CACHE_DIR, or LIQUIDIUM_CLI_DATA_DIR.");
}
const cacheDir = process.env.LIQUIDIUM_CLI_CACHE_DIR || join(process.env.XDG_CACHE_HOME || join(homeDir, ".cache"), "liquidium-borrow", "node");
const packageJsonPath = join(cacheDir, "package.json");
const packageLockPath = join(cacheDir, "package-lock.json");
const skillPackageJsonPath = join(skillDir, "package.json");
const skillPackageLockPath = join(skillDir, "package-lock.json");
const dataDir = process.env.LIQUIDIUM_CLI_DATA_DIR || join(process.env.XDG_DATA_HOME || join(homeDir, ".local", "share"), "liquidium-borrow");
const loansDir = join(dataDir, "loans");

const commands = new Set([
  "pools",
  "prices",
  "max-borrow",
  "quote",
  "instant-create",
  "instant-get",
  "instant-activities",
  "deposit-status",
  "instant-find",
  "loan-instructions",
  "repay-instructions",
  "add-collateral-instructions",
  "loan-list",
  "loan-show",
  "loan-tx",
  "profile-summary",
  "positions",
]);

function printHelp() {
  console.log(`Liquidium CLI

Usage:
  <skill-path>/scripts/liquidium-borrow <command> [options]

Commands:
  pools                         List Liquidium pools
  prices                        Get asset prices
  max-borrow                    Estimate max borrow amount for collateral
  quote                         Validate/preview LTV for an instant loan
  instant-create                Create an accountless instant loan
  instant-get                   Restore an instant loan by --ref or --loan-id
  instant-activities            List activity for an instant loan --ref
  deposit-status                Check whether Liquidium detected collateral deposit
  instant-find                  Find candidate instant loans by address
  loan-instructions             Show refreshed status/repay/add-collateral instructions
  repay-instructions            Shortcut for loan-instructions --action repay
  add-collateral-instructions   Shortcut for loan-instructions --action add-collateral
  loan-list                     List locally saved instant loan records
  loan-show                     Show a locally saved loan record by --ref
  loan-tx                       Append a collateral or repayment txid to a record
  profile-summary               Read aggregate profile portfolio summary
  positions                     List profile positions

Common options:
  --json                        Emit raw JSON instead of a compact summary
  --api-base-url URL            Override Liquidium SDK service URL
  --environment NAME            SDK environment, default mainnet
  --evm-rpc-url URL             EVM RPC URL for methods that need it
  --no-save                     Do not write/update local loan records

Examples:
  <skill-path>/scripts/liquidium-borrow pools
  <skill-path>/scripts/liquidium-borrow max-borrow --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005
  <skill-path>/scripts/liquidium-borrow quote --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005 --borrow-amount-decimal 9
  <skill-path>/scripts/liquidium-borrow instant-create --collateral-asset BTC --borrow-asset USDC --collateral-amount-decimal 0.0005 --borrow-amount-decimal 9 --borrow-destination 0x2222222222222222222222222222222222222222 --refund-destination bc1qrefunddestination
  <skill-path>/scripts/liquidium-borrow instant-get --ref 8Y9AQQ
  <skill-path>/scripts/liquidium-borrow deposit-status --ref 8Y9AQQ --txid <txid>
  <skill-path>/scripts/liquidium-borrow loan-instructions --ref 8Y9AQQ --action repay
  <skill-path>/scripts/liquidium-borrow loan-tx --ref 8Y9AQQ --kind collateral --txid <txid>
`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      args._.push(arg);
      continue;
    }

    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function ensureClientPackage() {
  mkdirSync(cacheDir, { recursive: true });
  if (existsSync(skillPackageJsonPath) && existsSync(skillPackageLockPath)) {
    copyFileSync(skillPackageJsonPath, packageJsonPath);
    copyFileSync(skillPackageLockPath, packageLockPath);
  } else if (!existsSync(packageJsonPath)) {
    throw new Error("Missing packaged Liquidium dependency manifest. Reinstall this skill.");
  }

  const requireFromCache = createRequire(packageJsonPath);
  try {
    return requireFromCache("@liquidium/client");
  } catch {
    const result = spawnSync("npm", ["ci", "--omit=dev", "--silent", "--ignore-scripts", "--no-audit", "--no-fund"], {
      cwd: cacheDir,
      stdio: "inherit",
    });

    if (result.status !== 0) {
      throw new Error("Failed to install @liquidium/client for the Liquidium CLI.");
    }

    return requireFromCache("@liquidium/client");
  }
}

function clientConfig(options) {
  const config = {};
  if (options.environment) config.environment = options.environment;
  if (options["api-base-url"] || process.env.LIQUIDIUM_API_BASE_URL) {
    config.apiBaseUrl = options["api-base-url"] || process.env.LIQUIDIUM_API_BASE_URL;
  }
  if (options["evm-rpc-url"] || process.env.LIQUIDIUM_EVM_RPC_URL) {
    config.evmRpcUrl = options["evm-rpc-url"] || process.env.LIQUIDIUM_EVM_RPC_URL;
  }
  if (options["timeout-ms"] || process.env.LIQUIDIUM_TIMEOUT_MS) {
    config.timeoutMs = Number(options["timeout-ms"] || process.env.LIQUIDIUM_TIMEOUT_MS);
    if (!Number.isFinite(config.timeoutMs) || config.timeoutMs <= 0) {
      throw new Error("--timeout-ms / LIQUIDIUM_TIMEOUT_MS must be a positive number");
    }
  }
  return config;
}

function parseBigIntOption(options, key, required = true) {
  const value = options[key];
  if (value === undefined || value === true) {
    if (required) throw new Error(`Missing --${key}`);
    return undefined;
  }
  if (!/^\d+$/.test(String(value))) {
    throw new Error(`--${key} must be an integer string in base units`);
  }
  return BigInt(value);
}

function decimalToBaseUnits(value, decimals) {
  const raw = String(value);
  if (!/^\d+(\.\d+)?$/.test(raw)) {
    throw new Error(`Decimal amount must be a non-negative number: ${raw}`);
  }
  const [whole, fraction = ""] = raw.split(".");
  const places = Number(decimals);
  if (fraction.length > places) {
    throw new Error(`Too many decimal places for asset decimals=${places}: ${raw}`);
  }
  const padded = fraction.padEnd(places, "0");
  return BigInt(whole || "0") * 10n ** BigInt(places) + BigInt(padded || "0");
}

function parseAmountOption(options, key, pool) {
  const baseValue = parseBigIntOption(options, key, false);
  if (baseValue !== undefined) return baseValue;
  const decimalValue = stringOption(options, `${key}-decimal`, false);
  if (decimalValue !== undefined) return decimalToBaseUnits(decimalValue, pool.decimals);
  throw new Error(`Missing --${key} or --${key}-decimal`);
}

function stringOption(options, key, required = true) {
  const value = options[key];
  if (value === undefined || value === true || value === "") {
    if (required) throw new Error(`Missing --${key}`);
    return undefined;
  }
  return String(value);
}

function bigintReplacer(_key, value) {
  return typeof value === "bigint" ? value.toString() : value;
}

function emitJson(value) {
  console.log(JSON.stringify(value, bigintReplacer, 2));
}

function safeFilePart(value) {
  return String(value).replace(/[^a-zA-Z0-9_.-]/g, "_");
}

function loanRecordPath(refOrLoanId) {
  return join(loansDir, `${safeFilePart(refOrLoanId)}.json`);
}

function writeLoanRecord(record) {
  mkdirSync(loansDir, { recursive: true });
  const ref = record.loan?.ref || record.summary?.ref || record.ref || record.loan?.loanId || record.loan?.id;
  if (!ref) throw new Error("Cannot save loan record without ref or loan id");
  const path = loanRecordPath(ref);
  writeFileSync(path, JSON.stringify(record, bigintReplacer, 2));
  return path;
}

function addSecondsIso(iso, seconds) {
  const started = new Date(iso);
  if (Number.isNaN(started.getTime())) return null;
  return new Date(started.getTime() + Number(seconds) * 1000).toISOString();
}

function depositDeadline({ loan, record, createdAt }) {
  const anchor = record?.createdAt || createdAt;
  const windowSeconds =
    loan?.depositWindowSeconds ||
    record?.loan?.depositWindowSeconds ||
    record?.request?.depositWindowSeconds;
  if (!anchor || windowSeconds === undefined || windowSeconds === null) {
    return {
      depositWindowSeconds: windowSeconds || null,
      estimatedDepositDeadlineAt: null,
      estimated: true,
    };
  }
  return {
    depositWindowSeconds: windowSeconds,
    estimatedDepositDeadlineAt: addSecondsIso(anchor, windowSeconds),
    estimated: true,
  };
}

function readLoanRecord(ref) {
  const path = loanRecordPath(ref);
  if (!existsSync(path)) throw new Error(`No local loan record found for ${ref}`);
  return JSON.parse(readFileSync(path, "utf8"));
}

function readLoanRecordIfExists(ref) {
  try {
    return readLoanRecord(ref);
  } catch {
    return null;
  }
}

function listLoanRecords() {
  if (!existsSync(loansDir)) return [];
  return readdirSync(loansDir)
    .filter((name) => name.endsWith(".json"))
    .sort()
    .map((name) => {
      const path = join(loansDir, name);
      try {
        const record = JSON.parse(readFileSync(path, "utf8"));
        return {
          path,
          ref: record.summary?.ref || record.loan?.ref || record.ref,
          status: record.summary?.status || record.loan?.status,
          collateral: record.collateral,
          borrow: record.borrow,
          createdAt: record.createdAt,
          updatedAt: record.updatedAt,
        };
      } catch (error) {
        return {
          path,
          error: true,
          errorMessage: error.message,
        };
      }
    });
}

function mergeLoanRecord(ref, updates) {
  const existing = readLoanRecordIfExists(ref) || {};
  const merged = {
    ...existing,
    ...updates,
    createdAt: existing.createdAt || updates.createdAt,
    transactions: updates.transactions || existing.transactions,
    request: existing.request || updates.request,
    collateral: updates.collateral || existing.collateral,
    borrow: updates.borrow || existing.borrow,
    transferInstructions:
      updates.transferInstructions || existing.transferInstructions,
    risk: updates.risk || existing.risk,
    rate: updates.rate || existing.rate,
    loan: updates.loan || existing.loan,
    summary: updates.summary || existing.summary,
  };
  return writeLoanRecord(merged);
}

function targetText(target) {
  if (!target) return null;
  if (target.type === "nativeAddress") return target.address;
  if (target.type === "icrcAccount") return target.account;
  if (target.address) return target.address;
  if (target.account) return target.account;
  return JSON.stringify(target, bigintReplacer);
}

function toBigInt(value, fallback = 0n) {
  if (value === undefined || value === null) return fallback;
  return typeof value === "bigint" ? value : BigInt(String(value));
}

function minBigInt(...values) {
  return values.reduce((min, value) => (value < min ? value : min));
}

function formatBaseUnits(value, decimals) {
  const amount = toBigInt(value);
  const places = Number(decimals);
  const scale = 10n ** BigInt(places);
  const whole = amount / scale;
  const fraction = (amount % scale).toString().padStart(places, "0").replace(/0+$/, "");
  return fraction ? `${whole}.${fraction}` : whole.toString();
}

function formatBps(value) {
  const bps = Number(value);
  return `${(bps / 100).toFixed(2)}%`;
}

function formatScaledRatePercent(scaledRate, rateDecimals, fractionDigits = 2) {
  const rate = toBigInt(scaledRate);
  const scale = 10n ** toBigInt(rateDecimals);
  const displayScale = 10n ** BigInt(fractionDigits);
  const rounded = (rate * 100n * displayScale + scale / 2n) / scale;
  const whole = rounded / displayScale;
  const fraction = rounded % displayScale;
  return `${whole}.${fraction.toString().padStart(fractionDigits, "0")}%`;
}

function scaledRateToBps(rate, rateDecimals) {
  const value = toBigInt(rate);
  const scale = 10n ** toBigInt(rateDecimals);
  return (value * 10_000n + scale / 2n) / scale;
}

function poolRatioToBps(value) {
  const ratio = toBigInt(value);
  if (ratio <= 10_000n) return ratio;
  return scaledRateToBps(ratio, 27n);
}

function estimateLiquidationPrice({ ltv, collateralPool }) {
  const liquidationThresholdBps = poolRatioToBps(collateralPool.liquidationThreshold);
  const ltvBps = toBigInt(ltv.ltvBps);
  if (ltvBps <= 0n || liquidationThresholdBps <= 0n) return null;

  const collateralUsd = Number(ltv.collateralUsd) / 1e8;
  const collateralAmount = Number(ltv.collateralAmount);
  const decimals = Number(collateralPool.decimals);
  if (!Number.isFinite(collateralUsd) || !Number.isFinite(collateralAmount) || collateralAmount <= 0) {
    return null;
  }

  const currentPrice = collateralUsd / (collateralAmount / 10 ** decimals);
  const liquidationPrice = currentPrice * (Number(ltvBps) / Number(liquidationThresholdBps));
  return {
    collateralAsset: collateralPool.asset,
    collateralChain: collateralPool.chain,
    liquidationThresholdBps,
    liquidationThreshold: formatBps(liquidationThresholdBps),
    estimatedCurrentCollateralPriceUsd: Number(currentPrice.toFixed(2)),
    estimatedLiquidationCollateralPriceUsd: Number(liquidationPrice.toFixed(2)),
    note:
      "Estimated from current quote data. It changes as prices, debt, interest, and oracle data change.",
  };
}

function findPool(pools, asset, chain) {
  return pools.find((pool) => {
    if (pool.asset !== asset) return false;
    if (chain && pool.chain !== chain) return false;
    return !pool.frozen;
  });
}

async function loadMarket(client) {
  return Promise.all([client.market.listPools(), client.market.getAssetPrices()]);
}

async function buildInstantRequest(client, options) {
  const collateralAsset = stringOption(options, "collateral-asset");
  const borrowAsset = stringOption(options, "borrow-asset");
  const collateralChain = stringOption(options, "collateral-chain", false);
  const borrowChain = stringOption(options, "borrow-chain", false);
  const [pools, prices] = await loadMarket(client);
  const collateralPool = findPool(pools, collateralAsset, collateralChain);
  const borrowPool = findPool(pools, borrowAsset, borrowChain);

  if (!collateralPool) throw new Error(`No non-frozen collateral pool for ${collateralAsset}`);
  if (!borrowPool) throw new Error(`No non-frozen borrow pool for ${borrowAsset}`);

  const collateralAmount = parseAmountOption(options, "collateral-amount", collateralPool);
  const borrowAmount = parseAmountOption(options, "borrow-amount", borrowPool);

  const ltv = client.quote.calculateLtv(
    {
      collateralPoolId: collateralPool.id,
      borrowPoolId: borrowPool.id,
      collateralAmount,
      borrowAmount,
    },
    pools,
    prices
  );

  const validationErrors = ltv.validationErrors || [];
  return {
    pools,
    prices,
    collateralPool,
    borrowPool,
    ltv,
    validationErrors,
    request: {
      collateralPoolId: collateralPool.id,
      borrowPoolId: borrowPool.id,
      collateralAsset,
      borrowAsset,
      collateralAmount,
      borrowAmount,
      ltvMaxBps: parseBigIntOption(options, "ltv-max-bps", false) || ltv.maxAllowedLtvBps,
      depositWindowSeconds:
        parseBigIntOption(options, "deposit-window-seconds", false) || 3_600n,
      borrowDestination: {
        type: "External",
        address: stringOption(options, "borrow-destination"),
      },
      refundDestination: {
        type: "External",
        address: stringOption(options, "refund-destination"),
      },
    },
  };
}

function summarizeLoan(loan) {
  return {
    ref: loan.ref,
    loanId: loan.loanId || loan.id,
    status: loan.status,
    depositTarget: targetText(loan.depositTarget),
    repaymentAmount: loan.repayment?.amount,
    repaymentTarget: targetText(loan.repayment?.target || loan.repayTarget),
    position: loan.position,
  };
}

function amountDisplayForLoanSide(side, amount) {
  if (!side?.decimals || amount === undefined || amount === null) return null;
  return formatBaseUnits(amount, side.decimals);
}

function buildLoanInstructions(loan, action, record = null) {
  const summary = summarizeLoan(loan);
  const collateral = record?.collateral;
  const borrow = record?.borrow;
  const transfer = record?.transferInstructions || {};
  const repayment = loan.repayment || {};
  const rate = record?.rate || {};
  const repayTarget = targetText(repayment.target || loan.repayTarget);
  const depositTarget = targetText(loan.depositTarget);
  const repayAmountDisplay = amountDisplayForLoanSide(
    borrow,
    repayment.amount
  );

  const base = {
    action,
    ref: summary.ref,
    loanId: summary.loanId,
    status: summary.status,
    collateral,
    borrow,
    currentPosition: loan.position,
    warning:
      "Liquidation risk: if your LTV reaches the liquidation threshold, collateral can be sold to repay the loan.",
    refundNote:
      "For instant loans, the refund address receives collateral returned after full repayment, or collateral refunds when funds cannot be applied, including failed/late/expired deposits.",
    localRecordFound: Boolean(record),
    depositDeadline: depositDeadline({ loan, record }),
    borrowRate: {
      expectedBorrowApy: rate.expectedBorrowApy || null,
      rawBorrowingRate: rate.rawBorrowingRate || null,
      rateDecimals: rate.rateDecimals || null,
    },
  };

  if (action === "repay") {
    return {
      ...base,
      instructions:
        "Send the borrowed asset to the repayment target. Refresh this instruction immediately before sending because interest accrues continuously.",
      repayAmountBaseUnits: repayment.amount,
      repayAmountDisplay,
      repayAsset: borrow?.asset || repayment.asset,
      repayChain: repayment.chain || borrow?.chain,
      repayTarget,
      fullRepaymentNote:
        "For instant loans, partial repayments reduce debt and may improve LTV, but they do not trigger collateral withdrawal. The loan must be repaid in full to receive the full collateral amount back.",
    };
  }

  if (action === "add-collateral") {
    return {
      ...base,
      instructions:
        "Send additional collateral asset to the collateral deposit target to lower LTV and improve loan health. This also works after the loan has started. Do not send repayment funds to this target.",
      collateralAsset: collateral?.asset || transfer.sendCollateralAsset,
      collateralChain: collateral?.chain || transfer.sendCollateralChain,
      depositTarget,
      originalCollateralAmount: transfer.sendCollateralAmount,
      originalCollateralAmountBaseUnits: transfer.sendCollateralAmountBaseUnits,
      nextStep:
        transfer.sendCollateralAmount && depositTarget
          ? `Send ${transfer.sendCollateralAmount} ${transfer.sendCollateralAsset || ""} to ${depositTarget}`.trim()
          : null,
    };
  }

  return {
    ...base,
    instructions: "Current refreshed loan state.",
    depositTarget,
    repayAmountBaseUnits: repayment.amount,
    repayAmountDisplay,
    repayTarget,
    nextStep:
      transfer.sendCollateralAmount && depositTarget
        ? `Send ${transfer.sendCollateralAmount} ${transfer.sendCollateralAsset || ""} to ${depositTarget}`.trim()
        : null,
  };
}

function summarizeMarketSide(pool, amount) {
  return {
    asset: pool.asset,
    chain: pool.chain,
    poolId: pool.id,
    decimals: pool.decimals,
    amountBaseUnits: amount,
    amountDisplay: formatBaseUnits(amount, pool.decimals),
  };
}

function buildDepositStatus({ loan, activities, record = null, txid = null }) {
  const summary = summarizeLoan(loan);
  const depositTarget = targetText(loan.depositTarget);
  const depositActivities = activities.filter(
    (activity) =>
      activity.direction === "inflow" && activity.kind === "deposit"
  );
  const matchingActivities = txid
    ? depositActivities.filter((activity) => {
        const txids = [
          activity.txid,
          ...(Array.isArray(activity.txids) ? activity.txids : []),
        ].filter(Boolean);
        return txids.includes(txid);
      })
    : [];
  const bestActivity =
    matchingActivities[0] ||
    depositActivities.find((activity) => activity.status !== "failed") ||
    depositActivities[0] ||
    null;
  const detected =
    depositActivities.length > 0 ||
    summary.status !== "awaiting_deposit" ||
    toBigInt(loan.position?.collateralAmount, 0n) > 0n;
  const processing =
    detected &&
    !["active", "settling", "closed"].includes(String(summary.status));

  return {
    ref: summary.ref,
    loanId: summary.loanId,
    loanStatus: summary.status,
    depositDetected: detected,
    depositProcessing: processing,
    depositTarget,
    expectedCollateral: record?.collateral || loan.collateral,
    depositDeadline: depositDeadline({ loan, record }),
    checkedTxid: txid,
    txidMatchedByLiquidium: txid ? matchingActivities.length > 0 : null,
    bestDepositActivity: bestActivity,
    depositActivities,
    confirmations: bestActivity
      ? {
          current: bestActivity.confirmations,
          required: bestActivity.requiredConfirmations,
        }
      : null,
    note: detected
      ? "Liquidium has detected a collateral deposit. Keep polling until the loan becomes active or closed."
      : "Liquidium has not surfaced a collateral deposit yet. It can take a few minutes after the Bitcoin transaction broadcasts or confirms before the SDK activity feed updates.",
  };
}

function summarizeLoanDetails(loan, built) {
  const summary = summarizeLoan(loan);
  const createdAt = new Date().toISOString();
  const borrowRate = built.borrowPool.borrowingRate
    ? formatScaledRatePercent(built.borrowPool.borrowingRate, built.borrowPool.rateDecimals)
    : null;
  return {
    summary,
    createdAt,
    depositDeadline: depositDeadline({ loan, createdAt }),
    collateral: summarizeMarketSide(built.collateralPool, built.request.collateralAmount),
    borrow: summarizeMarketSide(built.borrowPool, built.request.borrowAmount),
    risk: {
      currentLtvBps: built.ltv.ltvBps,
      currentLtv: formatBps(built.ltv.ltvBps),
      maxAllowedLtvBps: built.ltv.maxAllowedLtvBps,
      maxAllowedLtv: formatBps(built.ltv.maxAllowedLtvBps),
      liquidationThresholdBps: poolRatioToBps(
        built.collateralPool.liquidationThreshold
      ),
      liquidationThreshold: formatBps(
        poolRatioToBps(built.collateralPool.liquidationThreshold)
      ),
      liquidationEstimate: estimateLiquidationPrice({
        ltv: built.ltv,
        collateralPool: built.collateralPool,
      }),
      nearMaxLtv: Number(built.ltv.ltvBps) >= Number(built.ltv.maxAllowedLtvBps) * 0.9,
    },
    rate: {
      expectedBorrowApy: borrowRate,
      rawBorrowingRate: built.borrowPool.borrowingRate,
      rateDecimals: built.borrowPool.rateDecimals,
    },
    transferInstructions: {
      sendCollateralAmount: formatBaseUnits(built.request.collateralAmount, built.collateralPool.decimals),
      sendCollateralAmountBaseUnits: built.request.collateralAmount,
      sendCollateralAsset: built.collateralPool.asset,
      sendCollateralChain: built.collateralPool.chain,
      sendCollateralTo: summary.depositTarget,
      receiveBorrowAmount: formatBaseUnits(built.request.borrowAmount, built.borrowPool.decimals),
      receiveBorrowAmountBaseUnits: built.request.borrowAmount,
      receiveBorrowAsset: built.borrowPool.asset,
      receiveBorrowChain: built.borrowPool.chain,
      receiveBorrowTo: built.request.borrowDestination.address,
      refundCollateralTo: built.request.refundDestination.address,
      nextStep: `Send ${formatBaseUnits(built.request.collateralAmount, built.collateralPool.decimals)} ${built.collateralPool.asset} to ${summary.depositTarget}`,
    },
  };
}

async function buildMaxBorrow(client, options) {
  const collateralAsset = stringOption(options, "collateral-asset");
  const borrowAsset = stringOption(options, "borrow-asset");
  const collateralChain = stringOption(options, "collateral-chain", false);
  const borrowChain = stringOption(options, "borrow-chain", false);
  const [pools, prices] = await loadMarket(client);
  const collateralPool = findPool(pools, collateralAsset, collateralChain);
  const borrowPool = findPool(pools, borrowAsset, borrowChain);

  if (!collateralPool) throw new Error(`No non-frozen collateral pool for ${collateralAsset}`);
  if (!borrowPool) throw new Error(`No non-frozen borrow pool for ${borrowAsset}`);

  const collateralAmount = parseAmountOption(options, "collateral-amount", collateralPool);

  const rawAvailable = toBigInt(borrowPool.availableLiquidity, 0n);
  const borrowCapRemaining =
    borrowPool.borrowCap && borrowPool.totalDebt
      ? toBigInt(borrowPool.borrowCap) - toBigInt(borrowPool.totalDebt)
      : rawAvailable;
  let high = minBigInt(rawAvailable, borrowCapRemaining > 0n ? borrowCapRemaining : rawAvailable);
  let low = 0n;
  let best = 0n;
  let bestLtv = null;
  let maxAllowedLtvBps = parseBigIntOption(options, "target-ltv-bps", false);

  while (low <= high) {
    const mid = (low + high) / 2n;
    const ltv = client.quote.calculateLtv(
      {
        collateralPoolId: collateralPool.id,
        borrowPoolId: borrowPool.id,
        collateralAmount,
        borrowAmount: mid,
      },
      pools,
      prices
    );

    const validationErrors = ltv.validationErrors || [];
    const targetBps = maxAllowedLtvBps || toBigInt(ltv.maxAllowedLtvBps);
    if (validationErrors.length === 0 && toBigInt(ltv.ltvBps) <= targetBps) {
      best = mid;
      bestLtv = ltv;
      maxAllowedLtvBps = targetBps;
      low = mid + 1n;
    } else {
      high = mid - 1n;
    }
  }

  return {
    collateral: summarizeMarketSide(collateralPool, collateralAmount),
    borrow: summarizeMarketSide(borrowPool, best),
    maxBorrowAmountBaseUnits: best,
    maxBorrowAmountDisplay: formatBaseUnits(best, borrowPool.decimals),
    currentLtvBps: bestLtv?.ltvBps || "0",
    currentLtv: bestLtv ? formatBps(bestLtv.ltvBps) : "0.00%",
    maxAllowedLtvBps,
    maxAllowedLtv: formatBps(maxAllowedLtvBps || 0),
    liquidationThresholdBps: poolRatioToBps(collateralPool.liquidationThreshold),
    liquidationThreshold: formatBps(
      poolRatioToBps(collateralPool.liquidationThreshold)
    ),
    liquidationEstimate: bestLtv
      ? estimateLiquidationPrice({ ltv: bestLtv, collateralPool })
      : null,
    expectedBorrowApy: borrowPool.borrowingRate
      ? formatScaledRatePercent(borrowPool.borrowingRate, borrowPool.rateDecimals)
      : null,
    note: "This is based on current SDK market data and is not a guarantee of execution. Requote before creating a loan.",
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const command = options._[0];

  if (!command || options.help || options.h) {
    printHelp();
    return;
  }

  if (!commands.has(command)) {
    throw new Error(`Unknown command: ${command}`);
  }

  if (command === "loan-list") {
    return emitJson(listLoanRecords());
  }

  if (command === "loan-show") {
    const ref = stringOption(options, "ref");
    return emitJson(readLoanRecord(ref));
  }

  if (command === "loan-tx") {
    const ref = stringOption(options, "ref");
    const txid = stringOption(options, "txid");
    const kind = stringOption(options, "kind", false) || "collateral";
    const chain = stringOption(options, "chain", false);
    const record = readLoanRecord(ref);
    const entry = {
      kind,
      txid,
      chain,
      recordedAt: new Date().toISOString(),
    };
    record.transactions = [...(record.transactions || []), entry];
    record.updatedAt = entry.recordedAt;
    const path = writeLoanRecord(record);
    return emitJson({ path, transaction: entry });
  }

  const { LiquidiumClient, LiquidiumError } = ensureClientPackage();
  const client = new LiquidiumClient(clientConfig(options));

  try {
    if (command === "pools") {
      const pools = await client.market.listPools();
      if (options.json) return emitJson(pools);
      for (const pool of pools) {
        console.log(
          `${pool.asset}${pool.chain ? `/${pool.chain}` : ""} id=${pool.id} decimals=${pool.decimals} frozen=${Boolean(pool.frozen)}`
        );
      }
      return;
    }

    if (command === "prices") {
      return emitJson(await client.market.getAssetPrices());
    }

    if (command === "max-borrow") {
      return emitJson(await buildMaxBorrow(client, options));
    }

    if (command === "quote") {
      const built = await buildInstantRequest(client, {
        ...options,
        "borrow-destination": options["borrow-destination"] || "0x0000000000000000000000000000000000000000",
        "refund-destination": options["refund-destination"] || "bc1qplaceholder",
      });
      const result = {
        collateralPool: built.collateralPool,
        borrowPool: built.borrowPool,
        ltv: built.ltv,
        risk: {
          currentLtvBps: built.ltv.ltvBps,
          currentLtv: formatBps(built.ltv.ltvBps),
          maxAllowedLtvBps: built.ltv.maxAllowedLtvBps,
          maxAllowedLtv: formatBps(built.ltv.maxAllowedLtvBps),
          liquidationThresholdBps: poolRatioToBps(
            built.collateralPool.liquidationThreshold
          ),
          liquidationThreshold: formatBps(
            poolRatioToBps(built.collateralPool.liquidationThreshold)
          ),
          liquidationEstimate: estimateLiquidationPrice({
            ltv: built.ltv,
            collateralPool: built.collateralPool,
          }),
        },
        executable: built.validationErrors.length === 0,
        validationErrors: built.validationErrors,
      };
      return emitJson(result);
    }

    if (command === "instant-create") {
      const built = await buildInstantRequest(client, options);
      if (built.validationErrors.length > 0) {
        emitJson({ ok: false, validationErrors: built.validationErrors });
        process.exitCode = 2;
        return;
      }
      const loan = await client.instantLoans.create(built.request);
      const details = summarizeLoanDetails(loan, built);
      const record = {
        type: "instant-loan",
        createdAt: details.createdAt,
        updatedAt: details.createdAt,
        request: built.request,
        loan,
        ...details,
      };
      const path = options["no-save"] ? null : writeLoanRecord(record);
      if (options.json) return emitJson({ ...record, localRecordPath: path });
      const summary = details.summary;
      console.log(`ref: ${summary.ref}`);
      console.log(`local record: ${path || "not saved"}`);
      console.log(`status: ${summary.status}`);
      console.log(`deposit window seconds: ${details.depositDeadline.depositWindowSeconds ?? ""}`);
      console.log(`estimated deposit deadline: ${details.depositDeadline.estimatedDepositDeadlineAt ?? ""}`);
      console.log(`deposit amount: ${details.transferInstructions.sendCollateralAmount} ${details.transferInstructions.sendCollateralAsset} on ${details.transferInstructions.sendCollateralChain}`);
      console.log(`deposit target: ${details.transferInstructions.sendCollateralTo}`);
      console.log(`borrow amount: ${details.transferInstructions.receiveBorrowAmount} ${details.transferInstructions.receiveBorrowAsset} on ${details.transferInstructions.receiveBorrowChain}`);
      console.log(`borrow destination: ${details.transferInstructions.receiveBorrowTo}`);
      console.log(`refund destination: ${details.transferInstructions.refundCollateralTo}`);
      console.log(`current ltv: ${details.risk.currentLtv} (max ${details.risk.maxAllowedLtv})`);
      console.log(`liquidation threshold: ${details.risk.liquidationThreshold}`);
      if (details.risk.liquidationEstimate) {
        console.log(`estimated liquidation ${details.risk.liquidationEstimate.collateralAsset} price: $${details.risk.liquidationEstimate.estimatedLiquidationCollateralPriceUsd}`);
      }
      console.log(`expected borrow apy: ${details.rate.expectedBorrowApy ?? ""}`);
      console.log(`repayment amount: ${summary.repaymentAmount ?? ""}`);
      console.log(`repayment target: ${summary.repaymentTarget ?? ""}`);
      console.log(`NEXT STEP: ${details.transferInstructions.nextStep}`);
      return;
    }

    if (command === "instant-get") {
      const ref = stringOption(options, "ref", false);
      const loanId = stringOption(options, "loan-id", false);
      if (!ref && !loanId) throw new Error("Pass --ref or --loan-id");
      const loan = await client.instantLoans.get(ref ? { ref } : { loanId: BigInt(loanId) });
      if (!options["no-save"]) {
        const summary = summarizeLoan(loan);
        mergeLoanRecord(summary.ref || ref || loanId, {
          type: "instant-loan",
          updatedAt: new Date().toISOString(),
          ref: summary.ref || ref || loanId,
          loan,
          summary,
        });
      }
      return options.json ? emitJson(loan) : emitJson(summarizeLoan(loan));
    }

    if (
      command === "loan-instructions" ||
      command === "repay-instructions" ||
      command === "add-collateral-instructions"
    ) {
      const ref = stringOption(options, "ref");
      const action =
        command === "repay-instructions"
          ? "repay"
          : command === "add-collateral-instructions"
            ? "add-collateral"
            : stringOption(options, "action", false) || "status";
      if (!["status", "repay", "add-collateral"].includes(action)) {
        throw new Error("--action must be status, repay, or add-collateral");
      }
      const loan = await client.instantLoans.get({ ref });
      const summary = summarizeLoan(loan);
      const existing = readLoanRecordIfExists(summary.ref || ref);
      const instructions = buildLoanInstructions(loan, action, existing);
      const path = options["no-save"]
        ? null
        : mergeLoanRecord(summary.ref || ref, {
            type: "instant-loan",
            updatedAt: new Date().toISOString(),
            ref: summary.ref || ref,
            loan,
            summary,
            latestInstructions: instructions,
          });
      return emitJson({ ...instructions, localRecordPath: path });
    }

    if (command === "instant-activities") {
      const ref = stringOption(options, "ref");
      const filter = stringOption(options, "filter", false);
      const activities = await client.activities.list({
        shortRef: ref,
        ...(filter ? { filter } : {}),
      });
      return emitJson(activities);
    }

    if (command === "deposit-status") {
      const ref = stringOption(options, "ref");
      const txid = stringOption(options, "txid", false);
      const [loan, activities] = await Promise.all([
        client.instantLoans.get({ ref }),
        client.activities.list({ shortRef: ref, filter: "all" }),
      ]);
      const summary = summarizeLoan(loan);
      const record = readLoanRecordIfExists(summary.ref || ref);
      const status = buildDepositStatus({ loan, activities, record, txid });
      const updates = {
        type: "instant-loan",
        updatedAt: new Date().toISOString(),
        ref: summary.ref || ref,
        loan,
        summary,
        latestDepositStatus: status,
      };
      if (txid) {
        const existingTransactions =
          record && Array.isArray(record.transactions) ? record.transactions : [];
        const hasTxid = existingTransactions.some(
          (transaction) =>
            transaction.txid === txid && (transaction.kind || "collateral") === "collateral"
        );
        updates.transactions = hasTxid
          ? existingTransactions
          : [
              ...existingTransactions,
              {
                kind: "collateral",
                txid,
                recordedAt: new Date().toISOString(),
              },
            ];
      }
      const path = options["no-save"]
        ? null
        : mergeLoanRecord(summary.ref || ref, updates);
      return emitJson({ ...status, localRecordPath: path });
    }

    if (command === "instant-find") {
      const address = stringOption(options, "address");
      const candidates = await client.instantLoans.findByAddress(address);
      return emitJson(candidates);
    }

    if (command === "profile-summary") {
      const profileId = stringOption(options, "profile-id");
      const [summary, reserves, healthFactor] = await Promise.all([
        client.positions.getUserPositionSummary(profileId),
        client.positions.getUserReserves(profileId),
        client.positions.getHealthFactor(profileId),
      ]);
      return emitJson({ summary, reserves, healthFactor });
    }

    if (command === "positions") {
      const profileId = stringOption(options, "profile-id");
      return emitJson(await client.positions.listPositions(profileId));
    }
  } catch (error) {
    if (LiquidiumError && error instanceof LiquidiumError) {
      throw new Error(error.message);
    }
    throw error;
  }
}

main().catch((error) => {
  console.error(`error: ${error.message}`);
  process.exit(1);
});
