DEBUG=0
COMPRESS='com.hadoop.compression.lzo.LzopCodec'
# COMPRESS=None

MARK='0' # '0' for tuwen '1' for video '2' for both
TYPE='tuwen'

TUWEN_MARK='0'
VIDEO_MARK='1'
ALL_MARK='2'

HASH_ONLY=0
# in you app you can choose 2kw,3kw,5kw,1y,2y
FEAT_DIM=600000000 
ENCODING='gb18030'

FILTER_RATIO=0.

NUM_PRES=18

# mid, docid, click, product, abtestid, show_time, unlike, all_interest_cnt, ty, dur, user_show, ori_lr_score, lr_score
MID=0
DOCID=1
CLICK=2
PRODUCT=3
ABTESTID=4
SHOW_TIME=5
UNLIKE=6
ALL_INTEREST_CNT=7
TY=8
DUR=9
USER_SHOW=10
ORI_LR_SCORE=11
LR_SCORE=12
POSITION=13
VIDEO_TIME=14
IMPRESSION_TIME=15
TOKENS=16
JSON_DATA=17

# Keys in JSON_DATA
ARTICLE_PT='article_page_time'
TW_HISTORY='vd_history'
VD_HISTORY='tw_history'
REA='rea'
DEV_INFO='device_info'

# MIN_SHOW=5 # TODO remove min show, now just same as before
MIN_SHOW=3
MAX_SHOW=1000

MIN_FEAT_FREQ=10

MIN_FEAT_FREQ_HOUR=2

#store 1h mid-docid
STORE_MID_DOCID=0
#read 24h mid-dcoid
READ_MID_DOCID_HISTORY=0
