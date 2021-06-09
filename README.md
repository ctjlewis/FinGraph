# FinGraph

A simple abstraction layer on top of Matplotlib that allows for easily
displaying financial graphics.

# Examples

See `Example.ipynb`.

## GME and AMC returns

#### Code
```python
startEnd = ('2021-01-01', '2021-06-09', '1d')

GME = Stock("GME", *startEnd).returns()
AMC = Stock("AMC", *startEnd).returns()

FinPlot(
    title = "Return Profiles of GameStop and AMC",
    headingFont = "Public Sans",
    xlabel = "Date",
    ylabel = "Return",
    offset = (24,-12),
).multiplot(
    (GME, "GME", "black"),
    (AMC, "AMC", "red", (0,2)),
).save()
```

#### Saved at `FinGraphData/Return Profiles of GameStop and AMC.svg`

![](FinGraphData/Return%20Profiles%20of%20GameStop%20and%20AMC.svg)

## Profit Margins of Remote Work Firms

#### Code
```python
COLORS = {
    'UBER': '#000000',
    'FVRR': '#1DBF73',
    'LYFT': '#EA0B8C',
    'UPWK': '#37A000'
}

UPWK = (0.131, 3.0e8, -1.6659e7)
FVRR = (0.261, 1.06e8, -3.37363e7)
UBER = (0.220, 1.41e10, -8.506e9)
LYFT = (0.279, 3.62e9, -2.602e9)

# convert to %
fmt = [UPWK, FVRR, UBER, LYFT]
fmt = [(pair[0]*100, pair[1], pair[2]/pair[1]) for pair in fmt]

UPWK, FVRR, UBER, LYFT = fmt
    
FinPlot(
    "Profit Margins of Flexible Work Firms", 
    "Take Rate $T_R$", 
    "Revenue $R$",
    percentX=True,
    financialY=True,
).multiplot_point(
    (*UPWK, "Upwork", COLORS["UPWK"]),
    (*FVRR, "Fiverr", COLORS["FVRR"], (-1, 0)),
    (*UBER, "Uber", COLORS["UBER"], (0, -1)),
    (*LYFT, "Lyft", COLORS["LYFT"], (0, 1)),
    bubbleLabelTransform=lambda x: f"{round(x, 2)*100}%"
).save()
```

#### Saved at `FinGraphData/Profit Margins of Flexible Work Firms.svg`
![](FinGraphData/Profit%20Margins%20of%20Flexible%20Work%20Firms.svg)