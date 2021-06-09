import yfinance as yf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
from datetime import date
from pathvalidate import sanitize_filename
import shutil

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# most of the hard work
matplotlib.style.use('fivethirtyeight')

def human_readable(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def bool_mag(b):
    return 1 if b else -1

def make_index(tickers):
    assert(type(tickers) == list)

    stock = Stock(tickers[0])
    DATA = stock.get_data()

    for ticker in tickers[1:]:
      stock = Stock(ticker)
      DATA += stock.get_data()
    
    return DATA.dropna()

class Stock: 
  def __init__(
    self, 
    ticker, 
    start = '', 
    end = '', 
    interval = '1d', 
    color = 'black', 
    labelOffset = (0,0)
  ):
    self.ticker = ticker

    self.start = start
    self.end = end
    self.interval = interval
    
    self.color = color
    self.labelOffset = labelOffset

    self.filedir = f"FinGraphData/stocks/{self.ticker}/{self.interval}"
    self.filestring = f"{self.filedir}/{self.start}_{self.end}.csv"

  # @staticmethod
  # def clear_cache():
  #   shutil.rmtree('./stocks')

  # Download new data, cache it
  def download_data(self):
    # retrieve from YFinance
    TICKER_DATA = yf.Ticker(self.ticker).history(
      start = self.start,
      end = self.end,
      interval = self.interval
    )
    
    # make sure directory is set up
    if not os.path.exists(self.filedir):
        os.makedirs(self.filedir)
    
    # cache
    TICKER_DATA.to_csv(self.filestring)
    return TICKER_DATA
    
  # Read from disk
  def read_data(self):
      TICKER_DATA = pd.read_csv(self.filestring)
      TICKER_DATA["Date"] = pd.to_datetime(TICKER_DATA["Date"])
      TICKER_DATA = TICKER_DATA.set_index("Date")
      
      return TICKER_DATA

  def is_cached(self):
      return os.path.isfile(self.filestring)

  def get_data(self):
      if (self.is_cached()):
          return self.read_data()
      
      else:
          print(f"${self.ticker}: Ticker history is not cached, downloading")
          return self.download_data()
      
  def normalize_close(self, table):
      return 100 * (table["Close"] / table["Close"][0] - 1)

  def get_returns(self):
      
      if type(self.ticker) == str:
          # do stuff if str ticker
          if self.ticker == "GIGX":
              TICKER_DATA = make_index(["UBER", "LYFT", "UPWK", "FVRR"])
              
          elif self.ticker == "FAANG":
              TICKER_DATA = make_index(["FB", "AAPL", "AMZN", "NFLX", "GOOG"])
              
          else: 
              TICKER_DATA = self.get_data()
      else:
          # accept raw data if not str
          TICKER_DATA = self.ticker
          
      return self.normalize_close(TICKER_DATA)

  def returns(self, interval = '1d'):
        return self.get_returns()
    
  def returnsSMA(self, SMA, interval = '1d'):
      return self.returns().rolling(SMA).mean().dropna()

class FinGraph:
    """
    Set up a plot with the given labels. Use .plot or .multiplot methods to add data.
    """
    def __init__(self, 
                 title = "", 
                 xlabel = "", 
                 ylabel = "",
                 start = '2020-09-01',
                 end = '2021-02-02',
                 headingFont="Work Sans",
                 bodyFont="Public Sans",
                 outerstyle="light", 
                 innerstyle="light",
                 financialX = False,
                 financialY = False,
                 percentX = False,
                 percentY = False,
                 humanX = False,
                 humanY = False,
                 innerPadding=(0,0),
                 scilimitsX=(0,0), 
                 scilimitsY=(0,0),
                 offset=(0,0),
                 maxBubbleSize=250,
                 grid=False
                ):
        
        self.title = title
        self.grid = grid

        self.start = start
        self.end = end

        self.maxBubbleSize = maxBubbleSize
        self.bubbles = []
        self.xlabel, self.ylabel = xlabel, ylabel
        self.scilimitsX, self.scilimitsY = scilimitsX, scilimitsY

        self.headingFont = headingFont
        self.bodyFont = bodyFont
        
        self.labels = {}
        self.bubbleLabels = {}
        
        self.white = "white"
        self.darkblue = "#002f6c"
        self.datetimeX = False
        
        self.financialX, self.financialY = financialX, financialY
        self.percentX, self.percentY = percentX, percentY
        self.humanX, self.humanY = humanX, humanY
 
        self.mvx, self.mvy = offset
        self.innerPadding = innerPadding
        
        if outerstyle == "light":
            self.background = self.white
            self.fontcolor = self.darkblue

        else:
            self.background = self.darkblue
            self.fontcolor = self.white
            
        if innerstyle == "light":
            self.plotbg = self.white
            self.linealpha = 0.5
            self.examplealpha = 0.2
        else:
            self.plotbg = self.darkblue
            self.linealpha = 0.5
            self.examplealpha = 0.3            
        
        plt.figure(figsize = (16,9)).patch.set_facecolor(self.background)        
        plt.title(f"\n{title}\n", fontfamily = self.headingFont, fontsize = 32, weight = "heavy", color=self.fontcolor)
        plt.xlabel(f"\n{xlabel}\n", fontfamily = self.headingFont, fontsize = 16, weight = "bold", color=self.fontcolor)
        plt.ylabel(f"\n{ylabel}\n", fontfamily = self.headingFont, fontsize = 16, weight = "bold", color=self.fontcolor)
        
    def plot_example_stocks(self, SMA = False, offset = (0,0)):
        SPY = Stock('SPY')
        if not SMA:
            self.plot(SPY.get_returns(), alpha=self.examplealpha, label="S&P500", offset=offset)
        else:
            self.plot(SPY.get_returns().rolling(SMA).mean().dropna(), alpha=self.examplealpha, label="S&P500", offset=offset)
        
        self.percentY = True
        return self
        
    def plot(self, data, label = " ", color = "grey", offset = (0,-1), alpha = 0.5, zorder=2):
        
        if isinstance(data.index, pd.DatetimeIndex):
            self.datetimeX = True
            
        if label != "" and label != " ":
            self.labels[label] = offset
            
        plt.plot(data, label=label, color=color, alpha=alpha, zorder=zorder)    
        return self
    
    def plot_point(self, x, y, z, 
                   label = " ", 
                   color = "grey", 
                   labelOffset=(0,-1), 
                   bubbleOffset=(0,0),
                   bubbleLabelTransform=lambda x: f"${human_readable(x)}",
                   alpha = 0.67, 
                   innerLabelSize=0,
                   zorder=3
                  ):
        
        if len(self.bubbles) > 0:
            size = np.sqrt(abs(z)/max([abs(bubble) for bubble in self.bubbles]))*self.maxBubbleSize
            size = max(size, 10)
        else:
            size = 1
            
        # STORE BUBBLE LABEL DATA FOR LATER
        if label != "" and label != " ":

            offX, offY = labelOffset
            offX, offY = size / 2 * offX, size / 2 * offY
            
            pad = 14
            
            if offX != 0:
                offX += pad if offX > 0 else -pad
            if offY != 0:
                offY += pad if offY > 0 else -pad
            
            self.labels[label] = (offX, offY)
            
            bubbleX, bubbleY = bubbleOffset
            bubbleX, bubbleY = size / 2 * bubbleX, size / 2 * bubbleY
            bubbleOffset = (bubbleX, bubbleY)
            
            self.bubbleLabels[label] = {}
            self.bubbleLabels[label]["text"] = bubbleLabelTransform(z)
            self.bubbleLabels[label]["size"] = innerLabelSize if innerLabelSize != 0 else size/5
            self.bubbleLabels[label]["xy"] = bubbleOffset
            self.bubbleLabels[label]["zorder"] = zorder
        
        # PLOT BUBBLE
        plt.plot(x, y, 'o', ms = size, label = label, color = color, alpha = alpha, zorder=zorder)
        
        return self
    
    def multiplot_point(self, 
                        *args,
                        bubbleLabelTransform=lambda x: f"${human_readable(x)}", 
                        innerPadding=(0.33,0.33),
                        zorder=3
                       ):
        
        self.innerPadding = innerPadding
        i = zorder
        
        for args_i in args:
            self.bubbles.append(args_i[2])
            
        for args_i in args:
            self.plot_point(*args_i, bubbleLabelTransform=bubbleLabelTransform, zorder=i)
            i += 1
            
        return self
        
    def multiplot(self, *args, zorder=3):

        # imply percent by default
        self.percentY = True
        
        i = zorder
        for args_i in args:
            self.plot(*args_i, zorder=i)
            i += 1
            
        return self
    
    def apply_styles(self):
        
        ax = plt.gca()
        lines = ax.lines
        
        for line in ax.lines:
            label = line.get_label()
            
            if label != "" and label != " ":
                
                offX, offY = self.labels[label] if label in self.labels else (0,0)
                
                if offX != 0:
                    offX += len(label) * 3 if offX > 0 else -(len(label) * 3)
                
                x,y = line.get_xydata()[-1]
                
                drawBubble = label in self.bubbleLabels
                
                if drawBubble:
                    bubbleLabel = self.bubbleLabels[label]
                    zorder = bubbleLabel['zorder'] 
                else:
                    zorder = plt.getp(line, 'zorder') + 1
                
                ax.annotate(
                    label, 
                    xy=(x, y), 
                    xytext=(self.mvx + offX, self.mvy + offY), 
                    color=line.get_color(), 
                    fontfamily=self.headingFont,
                    fontweight="heavy", 
                    alpha=0.7,
                    textcoords="offset pixels",
                    size=14, 
                    va="center",
                    ha="center",
                    zorder=zorder
                )
                
                if drawBubble:
                    ax.annotate(
                        bubbleLabel['text'], 
                        xy=(x, y), 
                        xytext=bubbleLabel['xy'], 
                        color="white", 
                        fontfamily=self.headingFont,
                        fontweight="heavy",
                        textcoords="offset pixels",
                        size=bubbleLabel['size'], 
                        va="center",
                        ha="center",
                        zorder=zorder
                    )
                
        if self.datetimeX:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        
        if self.percentX:
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            
        if self.percentY:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            
        fancyX = self.percentX or self.financialX or self.humanX or self.datetimeX
        fancyY = self.percentY or self.financialY or self.humanY
            
        if not fancyX:
            plt.ticklabel_format(axis='x', useMathText=True)

        if not fancyY:
            plt.ticklabel_format(axis='y', useMathText=True)

        if self.scilimitsX != (0,0):
            plt.ticklabel_format(axis='x', scilimits=self.scilimitsX)

        if self.scilimitsY != (0,0):
            plt.ticklabel_format(axis='y', scilimits=self.scilimitsY)

        ax.tick_params(axis="x", colors=self.fontcolor, pad=20, labelsize=12)
        ax.tick_params(axis="y", colors=self.fontcolor, pad=20, labelsize=12, labelright=True)
        ax.set_facecolor(self.plotbg)
        
        if self.innerPadding != (0,0):
            plt.margins(*self.innerPadding)
            
        if self.humanX:
            ax.set_xticklabels([human_readable(tick) for tick in ax.get_xticks()])

        if self.humanY:
            ax.set_yticklabels([human_readable(tick) for tick in ax.get_yticks()])

        if self.financialX:
            ticks = [f"${human_readable(tick)}" for tick in ax.get_xticks()]
            ax.set_xticklabels(ticks)

        if self.financialY:
            ticks = [f"${human_readable(tick)}" for tick in ax.get_yticks()]
            ax.set_yticklabels(ticks)
            
        make_square_axes(plt.gca())
        if not self.grid: plt.grid(None)
        
        return self

    def xlim(self, *args, **kwargs):
        plt.xlim(*args, **kwargs)
        return self
    
    def ylim(self, *args, **kwargs):
        plt.ylim(*args, **kwargs)
        return self
    
    def save(self):
        if not os.path.exists("FinGraphData"):
            os.makedirs("FinGraphData")
            
        self.apply_styles()
        fname = f"FinGraphData/{sanitize_filename(self.title)}.svg"
        
        if self.background == "white" or self.background == "#fff":
            plt.savefig(fname, format="svg", pad_inches=0.25, bbox_inches="tight", transparent=True)
        else:
            plt.savefig(fname, format="svg", facecolor=self.background, pad_inches=0.25, bbox_inches="tight")
            
        print(f"Saved to {fname}.")
        plt.show()
        
        return self
    
    def show(self):
        self.apply_styles()
        plt.show()
        return self

# STATIC STUFF

def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

WatermarkFont = {
    'name': 'Public-Sans',
    'color':  '#003c8f',
    'weight': 'medium',
    'size': 14,
}