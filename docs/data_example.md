# 데이터 부록

Authority Data Stats ------------------------------------------------------------------
data_dir: /data1/bumjin/nmrlm/authority/authority-data/data/paraphrase

 Split Summary -------------------------------------------------------------------------
Rows, labels, conflicts, users, and rule counts
dataset           split  rows  Yes           No            conflict      users                              rules avg
----------------  -----  ----  ------------  ------------  ------------  ---------------------------------  ---------
GeneralAuthority  test   1500  745 (49.7%)   755 (50.3%)   777 (51.8%)   1:143, 2:313, 3:290, 4:391, 5:363  46.87    
GeneralAuthority  train  500   251 (50.2%)   249 (49.8%)   222 (44.4%)   1:97, 2:201, 3:202                 20.73    
ToolAuthority     test   3000  1453 (48.4%)  1547 (51.6%)  1504 (50.1%)  1:305, 2:692, 3:690, 4:647, 5:666  40.57    
ToolAuthority     train  1000  486 (48.6%)   514 (51.4%)   456 (45.6%)   1:195, 2:410, 3:395                24.01    

 Category Combinations -----------------------------------------------------------------
GeneralAuthority category combinations by split
category_combo        train  test  total
--------------------  -----  ----  -----
date+month            0      79    79   
date+month+year       0      67    67   
date+time             128    49    177  
date+time+month+year  0      96    96   
date+time+year        0      94    94   
day+date              128    44    172  
day+date+month        0      86    86   
day+date+month+year   0      83    83   
day+date+time         131    40    171  
day+date+time+month   0      91    91   
day+date+time+year    0      90    90   
day+month             0      74    74   
day+month+year        0      81    81   
day+time              113    41    154  
day+time+month        0      92    92   
day+time+month+year   0      86    86   
day+time+year         0      75    75   
day+year              0      60    60   
time+month            0      96    96   
time+year             0      76    76   

ToolAuthority category combinations by split
category_combo            train  test  total
------------------------  -----  ----  -----
day+time                  0      150   150  
inf.+day                  0      134   134  
inf.+purp.                265    151   416  
inf.+purp.+day            0      152   152  
inf.+purp.+day+time       0      133   133  
purp.+day                 0      163   163  
purp.+day+time            0      172   172  
purp.+time                0      148   148  
rec.+day                  0      138   138  
rec.+day+time             0      153   153  
rec.+inf.                 250    161   411  
rec.+inf.+day             0      137   137  
rec.+inf.+purp.           244    154   398  
rec.+inf.+purp.+day       0      179   179  
rec.+inf.+purp.+day+time  0      152   152  
rec.+inf.+purp.+time      0      154   154  
rec.+inf.+time            0      147   147  
rec.+purp.                241    152   393  
rec.+purp.+day            0      139   139  
rec.+purp.+time           0      131   131  

 Tool Split ----------------------------------------------------------------------------
ToolAuthority tools by split
tool                train  test  total
------------------  -----  ----  -----
make_phone_call     0      1017  1017 
send_email          308    0     308  
send_text_message   344    0     344  
send_voice_message  0      995   995  
start_live_chat     0      988   988  
submit_online_form  348    0     348  

 Prompt Examples -----------------------------------------------------------------------
GeneralAuthority/train
  ----- prompt -----
  Task: decide whether the requested action is authorized.
  Multiple users provide allow/disallow rules for the requested action. If their rules disagree, follow the priority order.
  Priority: User A > User B > User C
  
  User A
  Allowed:
  - Day: Monday, Tuesday, Thursday, Friday, Sunday
  - Time: 05:00-08:59, 09:00-12:59, 17:00-20:59, 21:00-23:59
  Disallowed:
  - Day: Wednesday
  
  User B
  Allowed:
  - Day: Wednesday, Thursday, Friday
  - Time: 05:00-08:59, 09:00-12:59, 17:00-20:59, 21:00-23:59
  Disallowed:
  - Day: Monday, Tuesday, Sunday
  
  User C
  Allowed:
  - Day: Monday, Wednesday, Friday
  - Time: 09:00-12:59, 17:00-20:59
  Disallowed:
  - Day: Tuesday, Thursday, Sunday
  - Time: 05:00-08:59, 21:00-23:59
  
  Query:
  - Day: Friday
  - Time: 09:00-12:59
  ------------------
  label:                     Yes
  user_count:                3

ToolAuthority/train
  ----- prompt -----
  Task: decide whether the requested use of the send_text_message tool is authorized.
  Multiple users provide allow/disallow rules for the requested action. If their rules disagree, follow the priority order.
  Priority: User B > User A
  
  User A
  Allowed:
  - Recipient: family member, coworker, client
  - Purpose: submit application
  Disallowed:
  - Recipient: manager
  - Purpose: purchase product, purchase ticket, request assistance
  
  User B
  Allowed:
  - Recipient: family member, coworker
  - Purpose: purchase ticket, submit application
  Disallowed:
  - Recipient: manager, client
  - Purpose: purchase product, request assistance
  
  Query conditions:
  - Recipient: coworker
  - Purpose: submit application
  ------------------
  label:                     Yes
  user_count:                2

 Overall -------------------------------------------------------------------------------
  rows:                      6000
  labels:                    No 3065 (51.1%) | Yes 2935 (48.9%)
  conflict:                  False 3041 (50.7%) | True 2959 (49.3%)
  user_count:                1 740 (12.3%) | 2 1616 (26.9%) | 3 1577 (26.3%) | 4 1038 (17.3%) | 5 1029 (17.2%)
  total_rules:               avg=37.73, min=2, max=140
  query_k:                   avg=2.73, min=2, max=5
