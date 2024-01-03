import csv
import time
import json
import pickle
import xgbfir
import warnings
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import metrics
warnings.filterwarnings("ignore")

class XGB_process():

    def __init__(self):
        self.columns = [ '有無升等', 
                    '年資(用離職日+今天扣)', '年紀(用離職日+今天扣)',
                    '職級代碼',
                    '職類代碼',
                    '性別', '婚姻狀況',
                    '是否須打卡',
                    '在職狀態',
                    '員工識別碼',
                    '離職日',
                    '平均加班時數',
                    '累積訓練次數',
                    '今天之後有請過假',
                    '六個月累積事假次數', '三個月累積事假次數', '兩個月累積事假次數',
                    '六個月累積年假次數', '三個月累積年假次數', '兩個月累積年假次數',
                    '六個月累積非住院病假次數', '三個月累積非住院病假次數', '兩個月累積非住院病假次數',
                    '六個月累積非住院病假+事假次數', '三個月累積非住院病假+事假次數', '兩個月累積非住院病假+事假次數',
                    '六個月平均事假次數', '三個月平均事假次數', '兩個月平均事假次數', 
                    '六個月平均年假次數', '三個月平均年假次數', '兩個月平均年假次數', 
                    '六個月平均非住院病假次數', '三個月平均非住院病假次數', '兩個月平均非住院病假次數',
                    '六個月平均非住院病假+事假次數', '三個月平均非住院病假+事假次數', '兩個月平均非住院病假+事假次數',
                    '月考核只有G6',
                    '平均訓練次數',
                    '一整年平均加班時數', '六個月平均加班時數', '三個月平均加班時數', '兩個月平均加班時數',
                    '平均升等次數', '平均降等次數',
                    '無平均月考核',
                    '最近調薪幅度是A', '最近調薪幅度是B', '最近調薪幅度是C', '最近調薪幅度是D',
                    '平均調薪幅度為A的次數', '平均調薪幅度為B的次數', '平均調薪幅度為C的次數',
                    '平均月考核是否在[1,2)', '平均月考核是否在[2,3)', '平均月考核是否在[3,4)', '平均月考核是否在[4,5]', '平均半年考核是否在[1,2)', '平均半年考核是否在[2,3]',
                    '最近停年', '平均停年',
                    '薪資水位',
                    '經驗年資', '職前公司數'
                    ]
        
        self.columns_Pred = [ '有無升等', 
            '年資(用離職日+今天扣)', 
            '年紀(用離職日+今天扣)',
            '職級代碼',
            '職類代碼',
            '性別', '婚姻狀況',
            '是否須打卡',
            '員工識別碼',
            '平均加班時數',
            '累積訓練次數',
            '今天之後有請過假',
            '六個月累積事假次數', '三個月累積事假次數', '兩個月累積事假次數',
            '六個月累積年假次數', '三個月累積年假次數', '兩個月累積年假次數',
            '六個月累積非住院病假次數', '三個月累積非住院病假次數', '兩個月累積非住院病假次數',
            '六個月累積非住院病假+事假次數', '三個月累積非住院病假+事假次數', '兩個月累積非住院病假+事假次數',
            '六個月平均事假次數', '三個月平均事假次數', '兩個月平均事假次數', 
            '六個月平均年假次數', '三個月平均年假次數', '兩個月平均年假次數', 
            '六個月平均非住院病假次數', '三個月平均非住院病假次數', '兩個月平均非住院病假次數',
            '六個月平均非住院病假+事假次數', '三個月平均非住院病假+事假次數', '兩個月平均非住院病假+事假次數',
            '月考核只有G6',
            '平均訓練次數',
            '一整年平均加班時數', '六個月平均加班時數', '三個月平均加班時數', '兩個月平均加班時數',
            '平均升等次數', '平均降等次數',
            '無平均月考核',
            '最近調薪幅度是A', '最近調薪幅度是B', '最近調薪幅度是C', '最近調薪幅度是D',
            '平均調薪幅度為A的次數', '平均調薪幅度為B的次數', '平均調薪幅度為C的次數',
            '平均月考核是否在[1,2)', '平均月考核是否在[2,3)', '平均月考核是否在[3,4)', '平均月考核是否在[4,5]', '平均半年考核是否在[1,2)', '平均半年考核是否在[2,3]',
            '最近停年', '平均停年',
            '薪資水位',
            '經驗年資', '職前公司數'
            ]
        
        self.ReplaceName = {
                        # personal information
                        '性別': 'sex', '年紀(用離職日+今天扣)': 'age',
                        '婚姻狀況': 'marriage', '年資(用離職日+今天扣)': 'job_tenure',
                        '職類代碼': 'work_type', 
                        '是否須打卡': 'clock_in', 
                        '職級代碼': 'level',
                        '在職狀態': 'in_work',
                        '員工識別碼': 'code',

                        # leave
                        '今天之後有請過假': 'leave_after_today',
                        '兩個月平均事假次數': 'personal_leave_60_count', '三個月平均事假次數': 'personal_leave_90_count', '六個月平均事假次數': 'personal_leave_180_count',
                        '兩個月平均年假次數': 'year_leave_60_count', '三個月平均年假次數': 'year_leave_90_count', '六個月平均年假次數': 'year_leave_180_count',
                        '兩個月平均非住院病假次數': 'sick_leave_60_count', '三個月平均非住院病假次數': 'sick_leave_90_count', '六個月平均非住院病假次數': 'sick_leave_180_count',
                        '六個月平均非住院病假+事假次數': 'sick_leave_personal_leave_180_count',
                        '三個月平均非住院病假+事假次數': 'sick_leave_personal_leave_90_count',
                        '兩個月平均非住院病假+事假次數': 'sick_leave_personal_leave_60_count',
                        '六個月累積事假次數': 'personal_leave_acc_6m', '三個月累積事假次數': 'personal_leave_acc_3m', '兩個月累積事假次數': 'personal_leave_acc_2m',
                        '六個月累積年假次數': 'year_leave_acc_6m', '三個月累積年假次數': 'year_leave_acc_3m', '兩個月累積年假次數': 'year_leave_acc_2m',
                        '六個月累積非住院病假次數': 'sick_leave_acc_6m', '三個月累積非住院病假次數': 'sick_leave_acc_3m', '兩個月累積非住院病假次數': 'sick_leave_acc_2m',
                        '六個月累積非住院病假+事假次數': 'sick_leave_personal_leave_acc_6m', 
                        '三個月累積非住院病假+事假次數': 'sick_leave_personal_leave_acc_3m', 
                        '兩個月累積非住院病假+事假次數': 'sick_leave_personal_leave_acc_2m',
                        # work overtime
                        '一整年平均加班時數': 'overtime_acc_1y', '六個月平均加班時數': 'overtime_acc_6m',
                        '三個月平均加班時數': 'overtime_acc_3m', '兩個月平均加班時數': 'overtime_acc_2m',
                        '平均加班時數': 'overtime_avg',
                    
                        # work examination
                        '月考核只有G6': 'month_examine_G6', '無平均月考核': 'no_month_examine',
                        '平均月考核是否在[1,2)': 'avg_month_examine_12', '平均月考核是否在[2,3)': 'avg_month_examine_23', '平均月考核是否在[3,4)': 'avg_month_examine_34',
                        '平均月考核是否在[4,5]': 'avg_month_examine_45', '平均半年考核是否在[1,2)': 'avg_year_examine_12', '平均半年考核是否在[2,3]': 'avg_year_examine_23',

                        # salary
                        '薪資水位': 'salary_level',
                        '最近調薪幅度是A': 'adj_range_A', '最近調薪幅度是B': 'adj_range_B',
                        '最近調薪幅度是C': 'adj_range_C', '最近調薪幅度是D': 'adj_range_D',
                        '平均調薪幅度為A的次數': 'adj_A_times',
                        '平均調薪幅度為B的次數': 'adj_B_times',
                        '平均調薪幅度為C的次數': 'adj_C_times',

                        # promotion
                        '有無升等': 'level_up',
                        '平均升等次數': 'level_up_times', '平均降等次數': 'level_down_times',
                        '最近停年': 'recently_stop_year', '平均停年': 'stop_year',

                        # # work experience
                        '經驗年資': 'before_job_tenure',
                        '職前公司數': 'before_company',
                        '平均訓練次數': 'training_times_avg',
                        '累積訓練次數': 'trainging_times_acc',
                        '離職日':'Resignation_day'
                        }
            
        self.ReplaceName_Pred = {
                # personal information
                '性別': 'sex', '年紀(用離職日+今天扣)': 'age', 
                '婚姻狀況': 'marriage', 
                '年資(用離職日+今天扣)': 'job_tenure', 
                '職類代碼': 'work_type', 
                '是否須打卡': 'clock_in', 
                '職級代碼': 'level',
                '員工識別碼': 'code',
                # leave
                '今天之後有請過假': 'leave_after_today',
                '兩個月平均事假次數': 'personal_leave_60_count', '三個月平均事假次數': 'personal_leave_90_count', '六個月平均事假次數': 'personal_leave_180_count',
                '兩個月平均年假次數': 'year_leave_60_count', '三個月平均年假次數': 'year_leave_90_count', '六個月平均年假次數': 'year_leave_180_count',
                '兩個月平均非住院病假次數': 'sick_leave_60_count', '三個月平均非住院病假次數': 'sick_leave_90_count', '六個月平均非住院病假次數': 'sick_leave_180_count',
                '六個月平均非住院病假+事假次數': 'sick_leave_personal_leave_180_count',
                '三個月平均非住院病假+事假次數': 'sick_leave_personal_leave_90_count',
                '兩個月平均非住院病假+事假次數': 'sick_leave_personal_leave_60_count',
                '六個月累積事假次數': 'personal_leave_acc_6m', '三個月累積事假次數': 'personal_leave_acc_3m', '兩個月累積事假次數': 'personal_leave_acc_2m',
                '六個月累積年假次數': 'year_leave_acc_6m', '三個月累積年假次數': 'year_leave_acc_3m', '兩個月累積年假次數': 'year_leave_acc_2m',
                '六個月累積非住院病假次數': 'sick_leave_acc_6m', '三個月累積非住院病假次數': 'sick_leave_acc_3m', '兩個月累積非住院病假次數': 'sick_leave_acc_2m',
                '六個月累積非住院病假+事假次數': 'sick_leave_personal_leave_acc_6m', 
                '三個月累積非住院病假+事假次數': 'sick_leave_personal_leave_acc_3m', 
                '兩個月累積非住院病假+事假次數': 'sick_leave_personal_leave_acc_2m',
                # work overtime
                '一整年平均加班時數': 'overtime_acc_1y', '六個月平均加班時數': 'overtime_acc_6m',
                '三個月平均加班時數': 'overtime_acc_3m', '兩個月平均加班時數': 'overtime_acc_2m',
                '平均加班時數': 'overtime_avg',
                # work examination
                '月考核只有G6': 'month_examine_G6', '無平均月考核': 'no_month_examine',
                '平均月考核是否在[1,2)': 'avg_month_examine_12', '平均月考核是否在[2,3)': 'avg_month_examine_23', '平均月考核是否在[3,4)': 'avg_month_examine_34',
                '平均月考核是否在[4,5]': 'avg_month_examine_45', '平均半年考核是否在[1,2)': 'avg_year_examine_12', '平均半年考核是否在[2,3]': 'avg_year_examine_23',

                # salary
                '薪資水位': 'salary_level',
                '最近調薪幅度是A': 'adj_range_A', '最近調薪幅度是B': 'adj_range_B',
                '最近調薪幅度是C': 'adj_range_C', '最近調薪幅度是D': 'adj_range_D',
                '平均調薪幅度為A的次數': 'adj_A_times',
                '平均調薪幅度為B的次數': 'adj_B_times',
                '平均調薪幅度為C的次數': 'adj_C_times',

                # promotion
                '有無升等': 'level_up',
                '平均升等次數': 'level_up_times', '平均降等次數': 'level_down_times',
                '最近停年': 'recently_stop_year', '平均停年': 'stop_year',

                # work experience
                '經驗年資': 'before_job_tenure',
                '職前公司數': 'before_company',
                '平均訓練次數': 'training_times_avg',
                '累積訓練次數': 'trainging_times_acc',
                }
        
        self.AUC_THRESHOLD_RATIO = 1
    def OnehotEncoding(self, featureName, data):
        tmp = pd.get_dummies(data[featureName])
        for col in tmp.columns:
            data[featureName+str(col)] = tmp[col]
        data = data.drop(featureName, axis=1)
        return data

    def CutList(self, origin, ratio):
        # return the ratio part of origin list
        ratio = 1-ratio
        i = 0
        result = []
        l = len(origin)*ratio
        for key in origin:
            i += 1
            if i > l:
                break
            result.append(key)
        return result
    
    def train(self, Today):
        # 使用日期當作檔案名稱
        GlobalName = Today
        data_name = f'Train_Dataset_{Today}.csv'
        data = pd.read_csv('data/' + data_name)

        ################
        # reading data #
        ################
        print('reading data...')

        data_selected = data[self.columns].rename(self.ReplaceName, axis='columns')
        data_selected = data_selected.drop(["code"], axis=1)
        print("【確認 feature 數量是否一致】")
        print('columns: ', len(self.columns))
        print('ReplaceName: ', len(self.ReplaceName))
        #################
        # preprocessing #
        #################
        print('preprocessing...')
        # 填補NAN之欄位
        for i in data_selected.index:
            # marriage
            if pd.isna(data_selected['marriage'][i]):
                if data_selected['age'][i] < 25:
                    data_selected['marriage'][i] = 1
                else:
                    data_selected['marriage'][i] = 2
        # salary level
        data_selected['salary_level'] = data_selected['salary_level'].fillna(data_selected['salary_level'].mean()) 
        # experience company & experience job tenure
        data_selected['before_company'] = data_selected['before_company'].fillna(0)  
        data_selected['before_job_tenure'] = data_selected['before_job_tenure'].fillna(0) 

        # mapping data(將字元轉為數字型式)
        data_selected['sex'] = data_selected['sex'].map({'M': 1, 'F': 0})
        data_selected['clock_in'] = data_selected['clock_in'].map({'Y': 1, 'N': 0})
        data_selected['marriage'] = data_selected['marriage'].map({1: 1, 2: 2, 3: 1})
        data_selected['work_type'] = data_selected['work_type'].map({'M': 1, 'S': 3, 'T': 4, 'X':5})
        data_selected['in_work'] = data_selected['in_work'].map({'A': 1, 'H': 2, 'M': 3, 'N': 4, 'Q': 0, 'S': 6, 'T': 7})

        # remove other working status(移除在職和離職以外之狀態)
        data_selected = data_selected.drop(data_selected[data_selected['in_work']==2].index)
        data_selected = data_selected.drop(data_selected[data_selected['in_work']==3].index)
        data_selected = data_selected.drop(data_selected[data_selected['in_work']==4].index)
        data_selected = data_selected.drop(data_selected[data_selected['in_work']==6].index)
        data_selected = data_selected.drop(data_selected[data_selected['in_work']==7].index)


        ######################################
        # condition setting(bounding people) 
        # 設定篩選人員條件
        ######################################
        print('condition setting...')
        print('     data shape before condition setting: ', data_selected.shape)
        data_selected = data_selected.drop('Resignation_day', axis=1)
        print('     data shape after condition setting: ', data_selected.shape)


        #######################
        # feature interaction #
        #######################
        print('starting feature interaction...')
        data_copy = data_selected.copy()
        for i in data_copy.columns:
            if type(data_copy[i][data_copy[i].index[0]]) == str:
                data_copy = data_copy.drop(i, axis=1)
        # 1.使用XGBOOST套件分析特徵
        xgb_rmodel = xgb.XGBRegressor().fit(data_copy.drop('in_work', axis=1), data_copy['in_work'])
        # 2.儲存分析結果(在相同目錄的feature interaction目錄底下)
        # 用xlsx格式儲存，可以用office查看
        file_name = GlobalName+'.xlsx'
        xgbfir.saveXgbFI(xgb_rmodel,  OutputXlsxFile='feature interaction/' + file_name)
        # 3.讀取分析結果
        FI_data = pd.read_excel('feature interaction/'+file_name, 'Interaction Depth 1')
        # 4.選取交集之特徵，若超過10種則取重要性前10名
        FI_num = FI_data.shape[0]
        if FI_num > 10:
            FI_num = 10# 最大選取交集特徵數量
        print('     choosed interaction features number: ', FI_num)
        # 5.產生新特徵
        for i in range(FI_num):
            FI_name = FI_data['Interaction'][i]
            # 分析結果的特徵名稱形式為：(特徵A)|(特徵B)，例如age|marriage(年齡和婚姻交集)
            # 因此需要取出個別名稱供計算使用
            features_name = FI_name.split('|')
            data_selected[FI_name] = data_selected[FI_name.split('|')[0]]    
            print('     '+str(i)+'...', end='')
            print(FI_name)
            # 把兩特徵做交集(相乘)
            for num in range(len(features_name)-1):
                for idx in data_selected.index:
                    data_selected[FI_name][idx] *= data_selected[features_name[num+1]][idx]


        ####################
        # one-hot encoding #
        ####################
        print('one hot encoding...')
        Onehot_list = ['in_work']
        for col in Onehot_list:
            print('    ', data_selected.shape)
            data_selected = self.OnehotEncoding(col, data_selected)

        # check nan
        for c in data_selected.columns:
            if data_selected[c].isna().any():
                print('NAN in: ', c)

        ###################################
        # find feature's importances rank 
        # 排名特徵重要性
        ###################################
        TEST_SIZE = 0.1   #testing set size
        TIMES = 10         #fit model times
        N_ESTIMATORS = 100
        MAX_DEPTH = 30

        importance_list = []
        importance_result = []
        print('generating importance list...')
        for times in range(TIMES):
            ############################################
            #####       split train test set       #####
            #####       分開訓練與測試資料集       #####
            ############################################
            data_shuffle = shuffle(data_selected)
            train = data_shuffle[int(TEST_SIZE*len(data_shuffle)):]
            test = data_shuffle[:int(TEST_SIZE*len(data_shuffle))]

            x_train = train.drop('in_work0', axis=1)
            x_train = x_train.drop('in_work1', axis=1)
            y_train = train[['in_work0', 'in_work1']]

            x_test = test.drop('in_work0', axis=1)
            x_test = x_test.drop('in_work1', axis=1)
            y_test = test[['in_work0', 'in_work1']]

            #################################
            #####       fit model       #####
            #################################
            # 使用random forest模型內部套件進行重要性排名
            forest = RandomForestClassifier(n_estimators = N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight='balanced')
            forest_fit = forest.fit(x_train, y_train)

            if not importance_list:
                importance_list = forest_fit.feature_importances_.tolist()
            else:
                for i in range(len(forest_fit.feature_importances_.tolist())):
                    importance_list[i] += forest_fit.feature_importances_.tolist()[i]

        importance_result = dict.fromkeys(x_train.columns)
        for name, importance in zip(x_train.columns, importance_list):
            importance_result[name] = importance/times 
        importance_result = ({k: v for k, v in sorted(importance_result.items(), key=lambda item: item[1])})

        # 加上特徵排名中文版
        importance_result_CH = importance_result.copy()
        for chineseName in list(self.ReplaceName.keys()):
            if self.ReplaceName[chineseName] in list(importance_result_CH.keys()):
                importance_result_CH[chineseName] = importance_result_CH.pop(self.ReplaceName[chineseName])
        importance_result_CH = ({k: v for k, v in sorted(importance_result_CH.items(), key=lambda item: item[1], reverse=True)})

        # 儲存特徵重要性排名之結果
        seconds = time.time()
        f = open('result/importance/'+time.ctime(seconds).replace(' ', '_').replace(':', '_')+'_'+data_name+".txt", "w", encoding='utf-8')
        f.write(json.dumps(importance_result_CH, ensure_ascii=False).encode('utf8').decode().replace(', ', ',\n'))
        f = open('result/importance/' + GlobalName + '.txt', "w", encoding='utf-8')
        f.write(json.dumps(importance_result, ensure_ascii=False).encode('utf8').decode().replace(', ', ',\n'))

        ############
        # Training #
        ############
        TRAIN_AND_TEST = False # 訓練 OR 訓練+測試(目前改為只有訓練版本)
        MODEL = 1 # 0: random forest, 1: xgboost(目前只有xgboost)

        print('Training...')
        date_copy = data_selected.copy()
        #############################################
        #####       choose >?% importance       #####
        #############################################
        # 選擇前X%重要性之特徵(最後版本是選擇所有特徵)
        print('     before choose % feature: ', date_copy.shape)
        del_list = self.CutList(importance_result, 1)
        for d in del_list:
            if d in date_copy.columns:
                date_copy = date_copy.drop(d, axis=1)
        print('     after choose % feature: ', date_copy.shape)

        for c in date_copy.columns:
            if date_copy[c].isna().any():
                print('NAN in: ', c)
        print('     in work: ', date_copy[date_copy['in_work1'] == 1].shape)
        print('     quit work: ', date_copy[date_copy['in_work1'] == 0].shape)
        if TRAIN_AND_TEST:
            pass
        else:
            #################
            # split X and Y #
            #################
            data_shuffle = shuffle(date_copy)
            X = data_shuffle.drop('in_work0', axis=1)
            X = X.drop('in_work1', axis=1)
            Y = data_shuffle[['in_work0', 'in_work1']]              
            if MODEL == 1:
                xgb_model = XGBClassifier(eval_metric='mlogloss', scale_pos_weight=4)
                xgb_model_fit = xgb_model.fit(X, Y['in_work0'])

                # 計算AUC threshold
                # 重新預測訓練資料以計算AUC threshold
                my_predict_proba = xgb_model_fit.predict_proba(X)
                fpr, tpr, threshold = metrics.roc_curve(Y['in_work0'], my_predict_proba[:,1], pos_label=1)
                thresh = threshold[np.argmax(tpr - fpr)]
                print('thresh of AUC:', thresh)
                c = 0
                for p in my_predict_proba:
                    if p[1] > thresh:
                        c += 1
                # 計算大於threshold之人數比例
                print('ratio of > AUC:', c/date_copy.shape[0])
                self.AUC_THRESHOLD_RATIO = c/date_copy.shape[0]

                with open('models/XGB_' + GlobalName + '.pkl', 'wb') as f:
                    pickle.dump(xgb_model_fit, f)

    def predict(self, Today):
        data_name = f'Predict_Dataset_{Today}.csv'
        data = pd.read_csv('data/' + data_name)

        data_selected = data[self.columns_Pred].rename(self.ReplaceName_Pred, axis='columns')
        data_copy = data_selected.copy()

        print('columns: ', len(self.columns_Pred))
        print('ReplaceName: ', len(self.ReplaceName_Pred))
        print('preprocessing...')
        
        # 填補NAN之欄位
        for i in data_selected.index:
            # marriage
            if pd.isna(data_selected['marriage'][i]):
                if data_selected['age'][i] < 25:
                    data_selected['marriage'][i] = 1
                else:
                    data_selected['marriage'][i] = 2

        # mapping data(將字元轉為數字型式)
        data_selected['sex'] = data_selected['sex'].map({'M': 1, 'F': 0})
        data_selected['clock_in'] = data_selected['clock_in'].map({'Y': 1, 'N': 0})
        data_selected['marriage'] = data_selected['marriage'].map({1: 1, 2: 2, 3: 1})
        data_selected['work_type'] = data_selected['work_type'].map({'M': 1, 'S': 3, 'T': 4, 'X':5})
        
        print(data_selected.columns)
        ######################################
        # condition setting(bounding people) 
        # 設定篩選人員條件
        ######################################
        print('condition setting...')
        print('     data shape before condition setting: ', data_selected.shape)
        code = data_selected['code']
        data_selected = data_selected.drop('code', axis=1)
        print('     data shape after condition setting: ', data_selected.shape)

        # global file name
        # 使用日期當作檔案名稱
        GlobalName = Today

        #######################
        # feature interaction #
        #######################
        print('starting feature interaction...')
        # 1.讀取分析結果(training時產生)
        FI_data = pd.read_excel('feature interaction/'+GlobalName+'.xlsx', 'Interaction Depth 1')
        FI_num = FI_data.shape[0]
        if FI_num > 10: FI_num = 10# 最大選取交集特徵數量
        print('     choosed interaction features number: ', FI_num)
        # 2.產生新特徵
        for i in range(FI_num):
            FI_name = FI_data['Interaction'][i]
            features_name = FI_name.split('|')
            data_selected[FI_name] = data_selected[FI_name.split('|')[0]]    
            print('     '+str(i)+'...', end='')
            print(FI_name)
            # 把兩特徵做交集(相乘)
            for num in range(len(features_name)-1):
                for idx in data_selected.index:
                    data_selected[FI_name][idx] *= data_selected[features_name[num+1]][idx]

        # 讀取重要性排名結果(training時產生)
        f = open('result/importance/' + GlobalName + '.txt', 'r')
        l = []
        Ratio = 1
        Ratio = 1-Ratio
        n = 0
        for line in f:
            l.append(line.strip().split(':')[0].split(':')[0].split('"')[1])
        for d in l:
            n += 1
            if n > len(l)*Ratio:
                break
            if d in data_selected.columns:
                data_selected = data_selected.drop(d, axis=1)
            
        # check nan
        for c in data_selected.columns:
            if data_selected[c].isna().any():
                print('NAN in: ', c)
        MODEL = 1 # 0: random forest, 1: xgboost

        # load model
        if MODEL == 1:
            model_name = 'XGB_' + GlobalName
            with open('models/' + model_name + '.pkl', 'rb') as f:
                model = pickle.load(f)

            AUC = True
            # 原先以AUC threshold形式決定預測結果(1 or 0)，最後改為輸出離職機率(1~0)
            if AUC:
                predict = np.array(model.predict_proba(data_selected))
                predict[predict>=0.5] = 1
                predict[predict<0.5] = 0
                
                predict = predict[:, 1]
            else:
                predict = model.predict(data_selected)
            predict_prob = np.array(model.predict_proba(data_selected))[:, 1]
            predict_prob = (predict_prob - np.min(predict_prob))/np.ptp(predict_prob)
            # 根據大於threshold人數比例的人數，計算最接近thresolh之機率
            n = int(self.AUC_THRESHOLD_RATIO*data_selected.shape[0])
            print('choosed count:', n)
            predict_prob_sorted = -np.sort(-predict_prob)
            thresh_prob = round(predict_prob_sorted[n], 4)
            print('thresh:', predict_prob_sorted[n])

            print('code: ', code.index)
            print('count predict0:', np.count_nonzero(predict == 0))
            print('count predict1:', np.count_nonzero(predict == 1))
            # 儲存預測結果
            if len(predict) == len(code):
                file_name = 'probability/XGB_'+GlobalName+'.csv'
                with open(file_name, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Employee_Id', 'Generate_Date', 'Turnover_Rate', 'Department_Code', 'Module'])
                    for i in range(len(predict)):

                        if data_copy[data_copy['code'] == code[code.index[i]]]['leave_after_today'].values[0] == 0:
                            writer.writerow([code[code.index[i]], str(datetime.datetime.now())[:19], predict_prob[i], 'DEPT', 'XGB'])
                        else:
                            writer.writerow([code[code.index[i]], str(datetime.datetime.now())[:19], 0, 'DEPT', 'XGB'])
            else:
                print('length of data and prediction doesn\'t match!')
                print(len(predict))
                print(len(code))

        save_path = 'probability/XGB_'+GlobalName+'.csv'
        return save_path, thresh_prob