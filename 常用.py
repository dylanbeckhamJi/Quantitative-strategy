from __future__ import print_function, absolute_import 
from gm.api import *
from gm.model.storage import context
import numpy as np
import pandas as pd
from datetime import  datetime,timezone, timedelta
import datetime
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  
pd.set_option('display.float_format', '{:.2f}'.format)
import warnings
warnings.filterwarnings('ignore')

#掘金量化
set_token('807b8ba88782d7343bfe3ad918f33f93a2610ee8')
account_id='28edb8c3-ed6d-11f0-9ac0-00163e022aa6'
