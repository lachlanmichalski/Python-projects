# Distance to default - Merton (1974)
Function to calculate Distance-to-default

Replication of dtd in R -> https://rdrr.io/rforge/ifrogs/src/R/dtd.R
Extra documentation -> https://rdrr.io/rforge/ifrogs/f/inst/doc/dtd.pdf

#Data can be retrieve directly from CRSP and COMPUSTAT for US firms

Requires subscription to WRDS with access to CRSP and COMPUSTAT, 
and create/have a free account with St Louis Fed FRED API 
(https://fred.stlouisfed.org/docs/api/fred/) 

ALL YOU NEED TO INPUT IS THE FOLLOWING:
    1.  List of permno codes
    2.  Start date
    3.  End date
