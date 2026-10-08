# CherryQ 10-invoice scaling checkpoint

The 10-invoice case is the first step beyond the fully verified five-invoice
hardware proof. It is still deliberately small enough for exact enumeration,
but large enough to show how the business search space and QUBO circuit
resources begin to grow.

## SME cash position

- cleared cash: £12,000
- protected payroll/tax/reserve: £7,000
- supplier-payment budget: £5,000
- budget unit: £500

## Invoices

| ID | Supplier | Category | Amount | Standalone defer loss |
| --- | --- | --- | ---: | ---: |
| A | Materials supplier | Revenue-critical | £1,500 | £100 |
| B | Delivery partner | Fulfilment-critical | £1,000 | £100 |
| C | Equipment hire | Business continuity | £1,000 | £350 |
| D | IT support | Business continuity | £500 | £100 |
| E | Packaging supplier | Fulfilment-critical | £1,500 | £250 |
| F | Key stock replenishment | Revenue-critical | £1,000 | £300 |
| G | Warehouse utilities | Business continuity | £1,000 | £400 |
| H | Priority courier capacity | Fulfilment-critical | £500 | £150 |
| I | Compliance software | Penalty-sensitive | £500 | £200 |
| J | Marketing services | Deferrable | £500 | £50 |

## Dependencies

- A + B: £900 additional loss if both are not paid.
- F + H: £700 additional loss if both are not paid.

The first dependency represents materials plus delivery for an active customer
order. The second represents priority stock plus the courier capacity needed to
turn that stock into near-term revenue.

## Classical truth

Exact enumeration and MILP are expected to agree on:

- pay: **A + B + F + G + H**
- spend: **£5,000**
- modelled loss: **£950**

The deliberately weak standalone loss-per-pound greedy rule chooses:

- C + D + F + G + H + I + J
- spend: £5,000
- modelled loss: £1,350
- gap to optimum: **£400**

This demonstrates why supplier dependencies matter.

## Scaling compared with five invoices

| Metric | 5 invoices | 10 invoices |
| --- | ---: | ---: |
| Business pay/defer combinations | 32 | 1,024 |
| QUBO binary variables incl. slack | 8 | 14 |
| QUBO state space | 256 | 16,384 |
| Quadratic couplings | 28 | 91 |
| p=1 logical CX estimate | 56 | 182 |

The QUBO resource count grows faster than the invoice count because the binary
slack penalty introduces dense pairwise couplings. This is important evidence:
larger business instances make both classical search and current QAOA circuits
harder. It is **not** evidence of quantum advantage.

## Commands

Classical / MILP / QUBO scaling benchmark:

```bash
python -m cherryq.scaling --json-out scaling-5-vs-10.json
```

Run the 10-invoice case without QAOA:

```bash
python -m cherryq.ten_invoice --json-out ten-invoice.json
```

Run the ideal statevector warm-start-X p=1 experiment:

```bash
python -m cherryq.ten_invoice \
  --qaoa \
  --maxiter 80 \
  --shots 2048 \
  --seed 42 \
  --relaxation-multistart 32 \
  --json-out ten-invoice-qaoa.json
```

Do not submit a Q20 job until the 14-qubit frozen circuit has been transpiled
offline and its physical depth/CZ count reviewed.
