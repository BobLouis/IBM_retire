import pandas as pd
from tbl_helper import TBL_process
from db_helper import DB_process
from xgb_helper import XGB_process
from nn_helper import NN_process
import sys
import datetime

mode = ''
TRAINING_DAY = str(datetime.datetime.now())[:10] 

print(sys.argv)
if len(sys.argv) > 1:
    if sys.argv[1] == 'SAVE' or sys.argv[1] == 'PREDICT':
        mode = sys.argv[1]
        TRAINING_DAY = sys.argv[2]


TODAY = TRAINING_DAY
SAVING_DAY = TODAY
TBL = TBL_process()
DB = DB_process()
XGB = XGB_process()
NN = NN_process()

if mode == 'SAVE':
    print('save mode')
    #########################################
    # Step 1: Create Train/ Predict Dataset 
    # 加工並儲存訓練/預測資料                
    #########################################

    # Read, Preprocess, and Save data
    BIG_SHEETS = DB.select_db_data()
    BIG_SHEETS_ENGINEERED = TBL.run(BIG_SHEETS, SAVING_DAY)
    TBL.save_big_sheet(BIG_SHEETS_ENGINEERED, SAVING_DAY)
    DB.update_module_status2(2)
elif mode == 'PREDICT':
    print('predict mode')
    THRESH_PROB_XGB = 0
    THRESH_PROB_NN = 0
    #########################
    # Step 1: Train XgBoost
    # 訓練XGBOOST 模型
    #########################
    DB.update_module_status(2)
    XGB.train(TRAINING_DAY)

    ##################################
    # Step 2: Use XgBoost to Predict 
    # 使用XGBOOST模型預測
    ##################################
    
    XGB_RESULT_PATH, THRESH_PROB_XGB = XGB.predict(TODAY)
    DB.update_module_status(3, THRESH_PROB_XGB, THRESH_PROB_NN)

    ################################
    # Step 3: Train Neural Network 
    # 訓練NN 模型
    ################################
    DB.update_module_status(4, THRESH_PROB_XGB, THRESH_PROB_NN)
    SCALER = NN.train(TRAINING_DAY)

    #########################################
    # Step 4: Use Neural Network to Predict 
    # 使用NN模型預測
    #########################################
    NN_RESULT_PATH, THRESH_PROB_NN = NN.predict(TODAY, SCALER)
    DB.update_module_status(5, THRESH_PROB_XGB, THRESH_PROB_NN)
    

    #############################
    # Step 5: Insert Data to DB 
    # 將結果寫回DB
    #############################
    DB.update_module_status(6, THRESH_PROB_XGB, THRESH_PROB_NN)

    # 將 XGB 預測資料匯入 DB
    DB.insert_result(XGB_RESULT_PATH)

    # 將 NN 預測資料匯入 DB
    DB.insert_result(NN_RESULT_PATH)
    