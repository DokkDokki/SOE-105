# Put your source data files here

The pipeline looks in this folder and finds the three tables by filename.
Both **.xlsx** and **.csv** work.

| Table | Filename must contain | Required? |
|---|---|---|
| Sales history | `sale`, `sales` or `order` | **Yes** |
| Item master | `item` or `product` | Recommended |
| Customer master | `customer` or `client` | Optional |

The files currently here are the originals supplied by Mitsubishi:
`Sale Order History.xlsx`, `Item.xlsx`, `Customer.xlsx`.

## Columns

Column names are matched loosely — extra spaces, different capitalisation and
small renames are handled automatically, and you are shown exactly how each
column was matched before anything runs.

**Sales history** — one row per order line
`Sales Order Date`, `Item Internal ID`, `Quantity` are essential.
`SO Internal ID`, `Sales Order Number`, `Sales Order Status`,
`Customer Internal ID`, `Company Name`, `Item Code`, `Base Price` are used if present.

**Item master**
`Item Internal ID`, `Item Code`, `Production Status`, `Product Category 1`,
`Product Category 2`, `List price`.

**Customer master**
`Customer Internal ID`, `Company Name`, `Customer Type`, `FA Sub Industries`.

## To refresh the forecast

Replace the files here and run:

```bash
python run_pipeline.py
```

Or use the **Upload new data** page in the dashboard, which does the same thing
without touching the command line.
