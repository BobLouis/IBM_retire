# # # # # # # # # # # # # # # # # # # #
# Step 01: Import Must-have Packages  #
# # # # # # # # # # # # # # # # # # # #
import pandas as pd
import numpy as np

class TBL_process():

    def __init__(self):
        pass

    def run(self, BIG_SHEETS, Today):
        try:
            big_sheet, ID_Day_N, ID_Exp_N, ID_Exp_P = self.tbl_01_preprocess(BIG_SHEETS[0])
            big_sheet = self.tbl_0203_preprocess(big_sheet, BIG_SHEETS[1], BIG_SHEETS[2], ID_Exp_N, ID_Exp_P, Today)
            big_sheet = self.tbl_0405_preprocess(big_sheet, BIG_SHEETS[3], BIG_SHEETS[4], ID_Exp_N, ID_Exp_P, Today)
            big_sheet = self.tbl_0607_preprocess(big_sheet, BIG_SHEETS[5], BIG_SHEETS[6], ID_Exp_N, ID_Exp_P, Today)
            big_sheet = self.tbl_0809_preprocess(big_sheet, BIG_SHEETS[7], BIG_SHEETS[8])
            big_sheet = self.tbl_1011_preprocess(big_sheet, BIG_SHEETS[9], BIG_SHEETS[10])
            big_sheet = self.tbl_1213_preprocess(big_sheet, BIG_SHEETS[11], BIG_SHEETS[12], ID_Exp_N, ID_Exp_P, Today)
            big_sheet = self.tbl_1415_preprocess(big_sheet, BIG_SHEETS[13], BIG_SHEETS[14], ID_Exp_N, ID_Exp_P, Today)
            big_sheet = self.tbl_1617_preprocess(big_sheet, BIG_SHEETS[15], BIG_SHEETS[16], ID_Exp_N, ID_Exp_P, Today)
            big_sheet = self.tbl_1819_preprocess(big_sheet, BIG_SHEETS[17], BIG_SHEETS[18], ID_Day_N)
            big_sheet = self.tbl_2021_preprocess(big_sheet, BIG_SHEETS[19], BIG_SHEETS[20], ID_Day_N)
            big_sheet = self.tbl_22_preprocess(big_sheet, BIG_SHEETS[21], Today)
            big_sheet = self.tbl_23_preprocess(big_sheet, BIG_SHEETS[22], ID_Day_N, Today)
            big_sheet = self.tbl_24_preprocess(big_sheet, BIG_SHEETS[23])
            big_sheet = self.operation_25_preprocess(big_sheet, Today)
            big_sheet = self.operation_26_preprocess(big_sheet, Today)
            big_sheet = self.operation_27_preprocess(big_sheet)
            big_sheet = self.operation_28_preprocess(big_sheet)
            big_sheet = self.operation_29_preprocess(big_sheet)
            big_sheet = self.operation_30_preprocess(big_sheet)
        except:
            print("Exception Occur when preprocessing tables.")
        
        return big_sheet

    
    def save_big_sheet(self, big_sheet, Today):
        ##########
        # 模式二 #
        ##########
        #移除期間以外離職員工
        for i in big_sheet.index:
            # print('index', i)
            if big_sheet["離職日"][i] != '':
                y = int(big_sheet["離職日"][i].split('/')[0])
                m = int(big_sheet["離職日"][i].split('/')[1])
                d = int(big_sheet["離職日"][i].split('/')[2])
                y_t = int(Today.split('-')[0])
                m_t = int(Today.split('-')[1])
                d_t = int(Today.split('-')[2])
                if y*10000 + m*100 + d < y_t*10000 + m_t*100 + d_t:
                    big_sheet = big_sheet.drop(i)
        ##########
        # 模式二 #
        ##########
        # 將 Training Data 儲存至 data/
        big_sheet.to_csv(f"data/Train_Dataset_{Today}.csv", encoding="utf-8-sig", index=False)
        # 將 Testing Data 儲存至 data/
        AT_WORK_INDEX = big_sheet["離職日"] == ""
        big_sheet[AT_WORK_INDEX].to_csv(f"data/Predict_Dataset_{Today}.csv", encoding="utf-8-sig", index=False)
        
    def tbl_01_preprocess(self, sheet):
        print(f"[資料前處理] Table 1：員工資料")
        
        # Drop columns won't be used
        sheet = sheet.drop(["name", "phone", "address"], axis=1)
        
        # marital_status 3 => 1
        sheet["marital_status"] = sheet.apply(lambda row: 1 if row["marital_status"] == 3 else row["marital_status"], axis = 1)
        
        # Additional info
        id_day_N = sheet[["employee_id", "turnover_date"]]
        id_exp_N = sheet[["employee_id", "seniority_date", "turnover_date"]]
        id_exp_P = sheet[["employee_id", "seniority_date"]]
        
        # Drop possibly duplicated rows
        unduplicate_index =  sheet["employee_id"].duplicated() == False
        sheet = sheet[unduplicate_index]    
        return sheet, id_day_N, id_exp_N, id_exp_P

    def tbl_0203_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Exp_N, ID_Exp_P, Today):

        print(f"[資料前處理] Table 2：離職員工於前一年的請假單")
        print(f"[資料前處理] Table 3：在職員工於前一年的請假單")
        
        for vacation in ["事假", "年假", "非住院病假"]:
            sheet_N, sheet_P = sheet_N_Ori, sheet_P_Ori
            
            # Drop unused column
            sheet_N, sheet_P = sheet_N.drop(["absence_end", "inserve_type"], axis=1), sheet_P.drop(["absence_end", "inserve_type"], axis=1)
            personal_leave_index_N, personal_leave_index_P = sheet_N["absence_type"] == vacation, sheet_P["absence_type"] == vacation
            sheet_N, sheet_P = sheet_N[personal_leave_index_N].reset_index(drop=True), sheet_P[personal_leave_index_P].reset_index(drop=True)
            
            # Calculate 離職人員's 在職時長
            sheet_N = pd.merge(sheet_N, ID_Exp_N, how="inner", on="employee_id")
            sheet_N["在職時長"] = sheet_N.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["seniority_date"])).days, axis=1)
            sheet_N["在職時長"] = sheet_N.apply(lambda row: row["在職時長"] if row["在職時長"] < 20*365 else 20*365, axis=1)
            sheet_N["在職時長"] = sheet_N.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
            sheet_N = sheet_N.drop(["seniority_date"], axis=1)
            
            # Calculate 在職人員's 在職時長
            sheet_P = pd.merge(sheet_P, ID_Exp_P, how="inner", on="employee_id")
            sheet_P["在職時長"] = sheet_P.apply(lambda row: (pd.to_datetime(Today) - pd.to_datetime(row["seniority_date"])).days, axis = 1)
            sheet_P["在職時長"] = sheet_P.apply(lambda row: row["在職時長"] if row["在職時長"] < 20*365 else 20*365, axis=1)
            sheet_P["在職時長"] = sheet_P.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
            sheet_P = sheet_P.drop(["seniority_date"], axis=1)
            
            # Get target df from 離職人員 in specific time range
            six_mon_index_N = pd.to_datetime(sheet_N["absence_start"]) > pd.to_datetime(sheet_N["turnover_date"]) - pd.DateOffset(days=180)
            thr_mon_index_N = pd.to_datetime(sheet_N["absence_start"]) > pd.to_datetime(sheet_N["turnover_date"]) - pd.DateOffset(days=90)
            two_mon_index_N = pd.to_datetime(sheet_N["absence_start"]) > pd.to_datetime(sheet_N["turnover_date"]) - pd.DateOffset(days=60)
            one_mon_index_N = pd.to_datetime(sheet_N["absence_start"]) > pd.to_datetime(sheet_N["turnover_date"]) - pd.DateOffset(days=30)

            sheet_6_mon_N = sheet_N[six_mon_index_N].reset_index(drop=True).drop(["turnover_date"], axis=1)
            sheet_3_mon_N = sheet_N[thr_mon_index_N].reset_index(drop=True).drop(["turnover_date"], axis=1)
            sheet_2_mon_N = sheet_N[two_mon_index_N].reset_index(drop=True).drop(["turnover_date"], axis=1)
            sheet_1_mon_N = sheet_N[one_mon_index_N].reset_index(drop=True).drop(["turnover_date"], axis=1)
            
            # Get target df from 在職人員 in specific time range
            six_mon_index_P = pd.to_datetime(sheet_P["absence_start"]) > pd.to_datetime(Today) - pd.DateOffset(days=180)
            thr_mon_index_P = pd.to_datetime(sheet_P["absence_start"]) > pd.to_datetime(Today) - pd.DateOffset(days=90)
            two_mon_index_P = pd.to_datetime(sheet_P["absence_start"]) > pd.to_datetime(Today) - pd.DateOffset(days=60)
            one_mon_index_P = pd.to_datetime(sheet_P["absence_start"]) > pd.to_datetime(Today) - pd.DateOffset(days=30)

            sheet_6_mon_P = sheet_P[six_mon_index_P].reset_index(drop=True)
            sheet_3_mon_P = sheet_P[thr_mon_index_P].reset_index(drop=True)
            sheet_2_mon_P = sheet_P[two_mon_index_P].reset_index(drop=True)
            sheet_1_mon_P = sheet_P[one_mon_index_P].reset_index(drop=True)
            
            # 將在職離職的資料合併在一起
            sheet_6_mon = pd.concat([sheet_6_mon_N, sheet_6_mon_P], axis=0).reset_index(drop=True)
            sheet_3_mon = pd.concat([sheet_3_mon_N, sheet_3_mon_P], axis=0).reset_index(drop=True)
            sheet_2_mon = pd.concat([sheet_2_mon_N, sheet_2_mon_P], axis=0).reset_index(drop=True)
            sheet_1_mon = pd.concat([sheet_1_mon_N, sheet_1_mon_P], axis=0).reset_index(drop=True)
            
            sheet_6_mon_count = sheet_6_mon.groupby(["employee_id"], as_index=False)["employee_id"].agg({f"六個月累積{vacation}次數":"count"})
            sheet_3_mon_count = sheet_3_mon.groupby(["employee_id"], as_index=False)["employee_id"].agg({f"三個月累積{vacation}次數":"count"})
            sheet_2_mon_count = sheet_2_mon.groupby(["employee_id"], as_index=False)["employee_id"].agg({f"兩個月累積{vacation}次數":"count"})
            sheet_1_mon_count = sheet_1_mon.groupby(["employee_id"], as_index=False)["employee_id"].agg({f"一個月累積{vacation}次數":"count"})

            # Create base column => use to calculate frequency
            sheet_stay_leave_N = sheet_N.groupby(["employee_id"], as_index=False)["在職時長"].agg({"在職時長": np.max})
            sheet_stay_leave_P = sheet_P.groupby(["employee_id"], as_index=False)["在職時長"].agg({"在職時長": np.max})
            sheet_stay_leave = pd.concat([sheet_stay_leave_N, sheet_stay_leave_P], axis=0).reset_index(drop=True)
            
            sheet_6_mon_count = pd.merge(sheet_6_mon_count, sheet_stay_leave, how="left", on="employee_id")
            sheet_3_mon_count = pd.merge(sheet_3_mon_count, sheet_stay_leave, how="left", on="employee_id")
            sheet_2_mon_count = pd.merge(sheet_2_mon_count, sheet_stay_leave, how="left", on="employee_id")
            sheet_1_mon_count = pd.merge(sheet_1_mon_count, sheet_stay_leave, how="left", on="employee_id")

            sheet_6_mon_count[f"六個月平均{vacation}次數"] = sheet_6_mon_count.apply(lambda row: row[f"六個月累積{vacation}次數"]/min(row["在職時長"], 180), axis=1)
            sheet_3_mon_count[f"三個月平均{vacation}次數"] = sheet_3_mon_count.apply(lambda row: row[f"三個月累積{vacation}次數"]/min(row["在職時長"],  90), axis=1)
            sheet_2_mon_count[f"兩個月平均{vacation}次數"] = sheet_2_mon_count.apply(lambda row: row[f"兩個月累積{vacation}次數"]/min(row["在職時長"],  60), axis=1)
            sheet_1_mon_count[f"一個月平均{vacation}次數"] = sheet_1_mon_count.apply(lambda row: row[f"一個月累積{vacation}次數"]/min(row["在職時長"],  30), axis=1)

            sheet_6_mon_count = sheet_6_mon_count.drop(["在職時長"], axis = 1)
            sheet_3_mon_count = sheet_3_mon_count.drop(["在職時長"], axis = 1)
            sheet_2_mon_count = sheet_2_mon_count.drop(["在職時長"], axis = 1)
            sheet_1_mon_count = sheet_1_mon_count.drop(["在職時長"], axis = 1)
            
            # 將 Feature Engineering 的 Feature 置入 big_sheet 當中
            big_sheet = pd.merge(big_sheet, sheet_6_mon_count, how="left", on="employee_id")
            big_sheet[[f"六個月累積{vacation}次數", f"六個月平均{vacation}次數"]] = big_sheet[[f"六個月累積{vacation}次數", f"六個月平均{vacation}次數"]].fillna(0)
            
            big_sheet = pd.merge(big_sheet, sheet_3_mon_count, how="left", on="employee_id")
            big_sheet[[f"三個月累積{vacation}次數", f"三個月平均{vacation}次數"]] = big_sheet[[f"三個月累積{vacation}次數", f"三個月平均{vacation}次數"]].fillna(0)

            big_sheet = pd.merge(big_sheet, sheet_2_mon_count, how="left", on="employee_id")
            big_sheet[[f"兩個月累積{vacation}次數", f"兩個月平均{vacation}次數"]] = big_sheet[[f"兩個月累積{vacation}次數", f"兩個月平均{vacation}次數"]].fillna(0)

            big_sheet = pd.merge(big_sheet, sheet_1_mon_count, how="left", on="employee_id")
            big_sheet[[f"一個月累積{vacation}次數", f"一個月平均{vacation}次數"]] = big_sheet[[f"一個月累積{vacation}次數", f"一個月平均{vacation}次數"]].fillna(0)

            # 延伸計算是否有請過事假
            big_sheet[f"六個月內有請過{vacation}"] = big_sheet.apply(lambda row: True if row[f"六個月累積{vacation}次數"] != 0 else False, axis=1)
            big_sheet[f"三個月內有請過{vacation}"] = big_sheet.apply(lambda row: True if row[f"三個月累積{vacation}次數"] != 0 else False, axis=1)
            big_sheet[f"兩個月內有請過{vacation}"] = big_sheet.apply(lambda row: True if row[f"兩個月累積{vacation}次數"] != 0 else False, axis=1)
            big_sheet[f"一個月內有請過{vacation}"] = big_sheet.apply(lambda row: True if row[f"一個月累積{vacation}次數"] != 0 else False, axis=1)
        
        # 延伸計算：將非住院病假和事假放在一起考慮
        DAYS = ["六個月", "三個月", "兩個月", "一個月"]
        for day in DAYS:
            big_sheet[f"{day}平均非住院病假+事假次數"] = big_sheet.apply(lambda row: row[f"{day}平均事假次數"] + row[f"{day}平均非住院病假次數"], axis=True)
            big_sheet[f"{day}累積非住院病假+事假次數"] = big_sheet.apply(lambda row: row[f"{day}累積事假次數"] + row[f"{day}累積非住院病假次數"], axis=True)
            big_sheet[f"{day}內有請過非住院病假+事假"] = big_sheet.apply(lambda row: True if row[f"{day}累積非住院病假+事假次數"] != 0 else False, axis=1)

        # 計算今天之後有請過假
        Sheet_N_After_Today, Sheet_P_After_Today = sheet_N_Ori, sheet_P_Ori
        
        Sheet_N_Unique = Sheet_N_After_Today.groupby(["employee_id"], as_index = False)["absence_start"].agg({"最近請假時間": np.max})
        Sheet_N_Unique["今天之後有請過假"] = Sheet_N_Unique.apply(lambda row: 1 if pd.to_datetime(row["最近請假時間"]) > pd.to_datetime(Today) else 0, axis=1)
        Sheet_N_Unique = Sheet_N_Unique.drop(["最近請假時間"], axis=1)

        Sheet_P_Unique = Sheet_P_After_Today.groupby(["employee_id"], as_index = False)["absence_start"].agg({"最近請假時間": np.max})
        Sheet_P_Unique["今天之後有請過假"] = Sheet_P_Unique.apply(lambda row: 1 if pd.to_datetime(row["最近請假時間"]) > pd.to_datetime(Today) else 0, axis=1)
        Sheet_P_Unique = Sheet_P_Unique.drop(["最近請假時間"], axis=1)
        
        Sheet_Unique = pd.concat([Sheet_N_Unique, Sheet_P_Unique], axis=0).reset_index(drop=True)

        big_sheet = pd.merge(big_sheet, Sheet_Unique, how="left", on="employee_id")
        big_sheet["今天之後有請過假"] = big_sheet["今天之後有請過假"].fillna(0)

        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_0405_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Exp_N, ID_Exp_P, Today):
        print(f"[資料前處理] Table 4：離職人員於離職前兩年的在職訓練次數")
        print(f"[資料前處理] Table 5：在職人員於離職前兩年的在職訓練次數")
        
        # 離職人員
        # 計算在職時間(同時處理20年 issue)
        sheet_N_Ori = pd.merge(sheet_N_Ori, ID_Exp_N, how="inner", on="employee_id")
        sheet_N_Ori["年資(用離職日+今天扣)"] = pd.to_datetime(sheet_N_Ori["turnover_date"]) - pd.to_datetime(sheet_N_Ori["seniority_date"])
        sheet_N_Ori["年資(用離職日+今天扣)"] = sheet_N_Ori.apply(lambda row: row["年資(用離職日+今天扣)"].days if row["年資(用離職日+今天扣)"].days < 20*365 else 20*365, axis=1)
        sheet_N_Ori["平均訓練次數"] = sheet_N_Ori.apply(lambda row: row["trainingtime"]/row["年資(用離職日+今天扣)"] if row["年資(用離職日+今天扣)"] != 0 else row["trainingtime"], axis = 1)
        sheet_N_Ori["累積訓練次數"] = sheet_N_Ori["trainingtime"]
        sheet_N_Ori = sheet_N_Ori.drop(["trainingtime", "seniority_date", "turnover_date", "年資(用離職日+今天扣)"], axis = 1)
        
        # 在職人員
        sheet_P_Ori = pd.merge(sheet_P_Ori, ID_Exp_P, how="inner", on="employee_id")
        sheet_P_Ori["年資(用離職日+今天扣)"] = pd.to_datetime(Today) - pd.to_datetime(sheet_P_Ori["seniority_date"])
        sheet_P_Ori["年資(用離職日+今天扣)"] = sheet_P_Ori.apply(lambda row: row["年資(用離職日+今天扣)"].days if row["年資(用離職日+今天扣)"].days < 20*365 else 20*365, axis=1)
        sheet_P_Ori["平均訓練次數"] = sheet_P_Ori.apply(lambda row: row["trainingtime"]/row["年資(用離職日+今天扣)"] if row["年資(用離職日+今天扣)"] != 0 else row["trainingtime"], axis = 1)
        sheet_P_Ori["累積訓練次數"] = sheet_P_Ori["trainingtime"]
        sheet_P_Ori = sheet_P_Ori.drop(["trainingtime", "seniority_date", "年資(用離職日+今天扣)"], axis = 1)

        # 合併兩張表
        sheet = pd.concat([sheet_N_Ori, sheet_P_Ori], axis=0).reset_index(drop=True)
        big_sheet = pd.merge(big_sheet, sheet, how="left", on = "employee_id")
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_0607_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Exp_N, ID_Exp_P, Today):
        print(f"[資料前處理] Table 6：離職員工於離職前一年的加班時數")
        print(f"[資料前處理] Table 7：在職員工於離職前一年的加班時數")
        
        sheet_N = pd.merge(sheet_N_Ori, ID_Exp_N, how="inner", on="employee_id")
        sheet_P = pd.merge(sheet_P_Ori, ID_Exp_P, how="inner", on="employee_id")
        ##############################
        # Step 1: 計算累積值/次數/平均值
        All_N_AGG = sheet_N.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({"累積加班時數(一整年)":np.sum})
        All_P_AGG = sheet_P.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({"累積加班時數(一整年)":np.sum})

        All_N_CNT = sheet_N.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({"加班次數(一整年)":"count"})
        All_P_CNT = sheet_P.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({"加班次數(一整年)":"count"})

        All_N_AVG = pd.merge(All_N_AGG, All_N_CNT, how="left", on="employee_id")
        All_N_AVG["平均加班時數"] = All_N_AVG.apply(lambda row: row["累積加班時數(一整年)"] / row["加班次數(一整年)"], axis=1)
        All_N_AVG = All_N_AVG.drop(["累積加班時數(一整年)", "加班次數(一整年)"], axis=1)

        All_P_AVG = pd.merge(All_P_AGG, All_P_CNT, how="left", on="employee_id")
        All_P_AVG["平均加班時數"] = All_P_AVG.apply(lambda row: row["累積加班時數(一整年)"] / row["加班次數(一整年)"], axis = 1)
        All_P_AVG = All_P_AVG.drop(["累積加班時數(一整年)", "加班次數(一整年)"], axis=1)

        # 合併兩張表
        All_AVG = pd.concat([All_N_AVG, All_P_AVG], axis=0).reset_index(drop=True)
        big_sheet = pd.merge(big_sheet, All_AVG, how="left", on="employee_id")
        big_sheet["平均加班時數"] = big_sheet["平均加班時數"].fillna(0)
        ############################
        # Step 2: 計算不同月份的統計值
        Sheet_05_Work_Overtime_N = pd.merge(sheet_N_Ori, ID_Exp_N, how="inner", on="employee_id")
        Sheet_06_Work_Overtime_P = pd.merge(sheet_P_Ori, ID_Exp_P, how="inner", on="employee_id")
        All_N = Sheet_05_Work_Overtime_N.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({"累積加班時數(一整年)":np.sum})
        All_P = Sheet_06_Work_Overtime_P.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({"累積加班時數(一整年)":np.sum})

        map_time = {180: "六個月", 90: "三個月", 60: "兩個月", 30: "一個月"}

        for TIME in [180, 90, 60, 30]:
            time_index_N   = pd.to_datetime(Sheet_05_Work_Overtime_N["attendance_date"]) > pd.to_datetime(Sheet_05_Work_Overtime_N["turnover_date"]) - pd.DateOffset(days=TIME)
            feature_N = Sheet_05_Work_Overtime_N[time_index_N].reset_index(drop=True) 
            feature_tot_N = feature_N.groupby(["employee_id"], as_index=False)["overtime_frequency"].agg({f"累積加班時數({map_time[TIME]})": np.sum })
            All_N = pd.merge(All_N, feature_tot_N, how="left", on="employee_id")

            time_index_P   = pd.to_datetime(Sheet_06_Work_Overtime_P["attendance_date"]) > pd.to_datetime(Today) - pd.DateOffset(days=TIME)
            feature_P = Sheet_06_Work_Overtime_P[time_index_P].reset_index(drop=True) 
            feature_tot_P = feature_P.groupby(["employee_id"], as_index = False)["overtime_frequency"].agg({f"累積加班時數({map_time[TIME]})": np.sum })
            All_P = pd.merge(All_P, feature_tot_P, how="left", on="employee_id")

        All_N = pd.merge(All_N, ID_Exp_N, how="inner", on="employee_id")
        All_N["在職時長"] = All_N.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["seniority_date"])).days, axis=1)
        All_N["在職時長"] = All_N.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
        All_N = All_N.drop(["seniority_date", "turnover_date"], axis=1)
        All_P = pd.merge(All_P, ID_Exp_P, how="inner", on="employee_id")
        All_P["在職時長"] = All_P.apply(lambda row: (pd.to_datetime(Today) -pd.to_datetime(row["seniority_date"])).days, axis=1)
        All_N["在職時長"] = All_P.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
        All_P = All_P.drop(["seniority_date"], axis=1)

        Sheet_05_Work_Overtime = pd.concat([All_N, All_P], axis=0).reset_index(drop=True).fillna(0)
        
        Sheet_05_Work_Overtime["一整年平均加班時數"] = Sheet_05_Work_Overtime.apply(lambda row: row[f"累積加班時數(一整年)"]/min(365, row["在職時長"]), axis=1)
        for TIME in [180, 90, 60, 30]:
            Sheet_05_Work_Overtime[f"{map_time[TIME]}平均加班時數"] = Sheet_05_Work_Overtime.apply(lambda row: row[f"累積加班時數({map_time[TIME]})"]/min(TIME, row["在職時長"]), axis = 1)

        big_sheet = pd.merge(big_sheet, Sheet_05_Work_Overtime, how="left", on="employee_id")
        cols = Sheet_05_Work_Overtime.columns[1:]
        big_sheet[cols] = big_sheet[cols].fillna(0)

        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    

        return big_sheet

    def tbl_0809_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori):
        print(f"[資料前處理] Table 8：離職員工職前經驗年資")
        print(f"[資料前處理] Table 9：在職員工職前經驗年資")
        sheet = pd.concat([sheet_N_Ori, sheet_P_Ori], axis=0).reset_index(drop=True)
        sheet["經驗年資"] = sheet["seniority"]
        sheet = sheet.drop(["seniority"], axis=1)
        
        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        big_sheet["經驗年資"] = big_sheet["經驗年資"].fillna(0)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def tbl_1011_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori):
        print(f"[資料前處理] Table 10：離職員工職前經驗公司數")
        print(f"[資料前處理] Table 11：在職員工職前經驗公司數")
        
        sheet = pd.concat([sheet_N_Ori, sheet_P_Ori], axis=0).reset_index(drop=True)
        sheet["職前公司數"] = sheet["num_of_company"]
        sheet = sheet.drop(["num_of_company"], axis=1)
        
        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        big_sheet["職前公司數"] = big_sheet["職前公司數"].fillna(0)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]        
        return big_sheet

    def tbl_1213_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Exp_N, ID_Exp_P, Today):
        print(f"[資料前處理] Table 12：離職員工升等次數")
        print(f"[資料前處理] Table 13：在職員工升等次數")
        
        # 離職人員
        sheet_N = pd.merge(sheet_N_Ori, ID_Exp_N, how="left", on="employee_id")
        sheet_N["在職時長"] = sheet_N.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["seniority_date"])).days, axis=1)
        sheet_N["在職時長"] = sheet_N.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
        sheet_N["在職時長"] = sheet_N.apply(lambda row: row["在職時長"] if row["在職時長"] <= 20*365 else 20*365, axis=1)

        # 在職人員
        sheet_P = pd.merge(sheet_P_Ori, ID_Exp_P, how="left", on="employee_id")
        sheet_P["在職時長"] = sheet_P.apply(lambda row: (pd.to_datetime(Today) - pd.to_datetime(row["seniority_date"])).days, axis=1)
        sheet_P["在職時長"] = sheet_P.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
        sheet_P["在職時長"] = sheet_P.apply(lambda row: row["在職時長"] if row["在職時長"] <= 20*365 else 20*365, axis=1)

        # 合併
        sheet = pd.concat([sheet_N, sheet_P], axis=0).reset_index(drop=True)
        sheet["平均升等次數"] = sheet.apply(lambda row: row["num_of_upgrade"]/row["在職時長"], axis = 1)
        sheet = sheet.drop(["seniority_date", "turnover_date", "在職時長", "num_of_upgrade"], axis = 1)

        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        big_sheet["平均升等次數"] = big_sheet["平均升等次數"].fillna(0)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_1415_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Exp_N, ID_Exp_P, Today):
        print(f"[資料前處理] Table 14：離職員工降等次數")
        print(f"[資料前處理] Table 15：在職員工降等次數")
        
        # 離職人員
        sheet_N = pd.merge(sheet_N_Ori, ID_Exp_N, how="left", on="employee_id")
        sheet_N["在職時長"] = sheet_N.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["seniority_date"])).days, axis=1)
        sheet_N["在職時長"] = sheet_N.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
        sheet_N["在職時長"] = sheet_N.apply(lambda row: row["在職時長"] if row["在職時長"] <= 20*365 else 20*365, axis=1)

        # 在職人員
        sheet_P = pd.merge(sheet_P_Ori, ID_Exp_P, how="left", on="employee_id")
        sheet_P["在職時長"] = sheet_P.apply(lambda row: (pd.to_datetime(Today) - pd.to_datetime(row["seniority_date"])).days, axis=1)
        sheet_P["在職時長"] = sheet_P.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)
        sheet_P["在職時長"] = sheet_P.apply(lambda row: row["在職時長"] if row["在職時長"] <= 20*365 else 20*365, axis=1)

        # 合併
        sheet = pd.concat([sheet_N, sheet_P], axis=0).reset_index(drop=True)
        sheet["平均降等次數"] = sheet.apply(lambda row: row["num_of_downgrade"]/row["在職時長"], axis=1)
        sheet = sheet.drop(["seniority_date", "turnover_date", "在職時長", "num_of_downgrade"], axis=1)

        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        big_sheet["平均降等次數"] = big_sheet["平均降等次數"].fillna(0)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_1617_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Exp_N, ID_Exp_P, Today):
        print(f"[資料前處理] Table 16：離職員工職等停年")
        print(f"[資料前處理] Table 17：在職員工職等停年")
        # 離職人員
        sheet_N = sheet_N_Ori
        AVG_N = sheet_N.groupby(["employee_id"], as_index=False)["suspend_year"].agg({"平均停年": np.mean}) # 取得平均停年info

        sheet_Reshape_N = sheet_N.groupby(["employee_id"], as_index=False)["job_grade"].agg({"最近職等": np.max })
        sheet_Reshape_N = pd.merge(sheet_Reshape_N, AVG_N, how="left", on="employee_id")
        sheet_Reshape_N = pd.merge(sheet_Reshape_N, sheet_N, how="left", left_on=["employee_id", "最近職等"], right_on=["employee_id", "job_grade"])

        sheet_Reshape_N["最近停年"] = sheet_Reshape_N["suspend_year"]
        sheet_Reshape_N = sheet_Reshape_N.drop(["最近職等", "job_grade", "suspend_year"], axis=1)

        # 在職人員
        sheet_P = sheet_P_Ori
        AVG_P = sheet_P.groupby(["employee_id"], as_index=False)["suspend_year"].agg({"平均停年": np.mean}) # 取得平均停年info

        sheet_Reshape_P = sheet_P.groupby(["employee_id"], as_index=False)["job_grade"].agg({"最近職等": np.max })
        sheet_Reshape_P = pd.merge(sheet_Reshape_P, AVG_P, how="left", on="employee_id")
        sheet_Reshape_P = pd.merge(sheet_Reshape_P, sheet_P, how="left", left_on=["employee_id", "最近職等"], right_on=["employee_id", "job_grade"])

        sheet_Reshape_P["最近停年"] = sheet_Reshape_P["suspend_year"]
        sheet_Reshape_P = sheet_Reshape_P.drop(["最近職等", "job_grade", "suspend_year"], axis=1)

        sheet = pd.concat([sheet_Reshape_N, sheet_Reshape_P], axis=0).reset_index(drop=True)

        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_1819_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Day_N):
        print(f"[資料前處理] Table 18：離職員工於離職前兩年的月考核")
        print(f"[資料前處理] Table 19：在職員工近前兩年的月考核")
        ###########
        # PART 01 # One-hot Every Assessment
        ###########
        ID = ID_Day_N.drop(["turnover_date"], axis=1)

        ##########
        # 離職人員
        sheet_N = sheet_N_Ori
        
        # 將 G6 丟掉
        Keep_index = sheet_N["assessment_grade"].isin(["G1", "G2", "G3", "G4", "G5"])
        sheet_N = sheet_N[Keep_index].reset_index(drop=True)
        
        # 計算平均月考核(Only 考慮有 G1~G5 的那群人)
        sheet_N["月考核"] = sheet_N.apply(lambda row: int(row["assessment_grade"][1]), axis=1)
        AVG_N = sheet_N.groupby(["employee_id"], as_index=False)["月考核"].agg({"平均月考核": np.mean})

        # 取得最近月考核(Only 考慮有G1~G5的那群人)
        sheet_Reshape_N = sheet_N.groupby(["employee_id"], as_index=False)["period"].agg({"最近期間":np.max})
        sheet_Reshape_N = pd.merge(sheet_Reshape_N, AVG_N, how="left", on="employee_id")
        sheet_Reshape_N = pd.merge(sheet_Reshape_N, sheet_N, how="left", left_on=["employee_id", "最近期間"], right_on=["employee_id", "period"])
        sheet_Reshape_N["最近月考核"] = sheet_Reshape_N["月考核"]
        
        # Drop
        sheet_Reshape_N = sheet_Reshape_N.drop(["最近期間", "period", "assessment_grade", "月考核"], axis=1)
        
        ##########
        # 在職人員
        sheet_P = sheet_P_Ori

        # 將 G6 丟掉
        Keep_index = sheet_P["assessment_grade"].isin(["G1", "G2", "G3", "G4", "G5"])
        sheet_P = sheet_P[Keep_index].reset_index(drop=True)
        
        # 計算平均月考核(Only 考慮有G1~G5的那群人)
        sheet_P["月考核"] = sheet_P.apply(lambda row: int(row["assessment_grade"][1]), axis = 1)
        AVG_P = sheet_P.groupby(["employee_id"], as_index = False)["月考核"].agg({"平均月考核": np.mean})

        # 取得最近月考核(Only 考慮有G1~G5的那群人)
        sheet_Reshape_P = sheet_P.groupby(["employee_id"], as_index=False)["period"].agg({"最近期間":np.max})
        sheet_Reshape_P = pd.merge(sheet_Reshape_P, AVG_P, how="left", on="employee_id")
        sheet_Reshape_P = pd.merge(sheet_Reshape_P, sheet_P, how="left", left_on=["employee_id", "最近期間"], right_on=["employee_id", "period"])
        sheet_Reshape_P["最近月考核"] = sheet_Reshape_P["月考核"]

        # Drop
        sheet_Reshape_P = sheet_Reshape_P.drop(["最近期間", "period", "assessment_grade", "月考核"], axis=1)

        #############
        # 合併
        sheet = pd.concat([sheet_Reshape_N, sheet_Reshape_P], axis=0).reset_index(drop=True)
        sheet = pd.merge(ID, sheet, how="left", on="employee_id")
        
        # 製作：有無月考核/平均月考核是否在[1,2)/平均月考核是否在[2,3)/平均月考核是否在[3,4)/平均月考核是否在[4,5]
        sheet["無平均月考核"] = sheet.apply(lambda row: 1 if np.isnan(row["平均月考核"]) else 0, axis=1)
        sheet["平均月考核是否在[1,2)"] = sheet.apply(lambda row: 1 if (row["平均月考核"] >= 1 and row["平均月考核"] < 2) else 0, axis=1)
        sheet["平均月考核是否在[2,3)"] = sheet.apply(lambda row: 1 if (row["平均月考核"] >= 2 and row["平均月考核"] < 3) else 0, axis=1)
        sheet["平均月考核是否在[3,4)"] = sheet.apply(lambda row: 1 if (row["平均月考核"] >= 3 and row["平均月考核"] < 4) else 0, axis=1)
        sheet["平均月考核是否在[4,5]"] = sheet.apply(lambda row: 1 if (row["平均月考核"] >= 4 and row["平均月考核"] <= 5) else 0, axis=1)

        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        ###########
        # PART 02 # 只有 G6
        ###########
        sheet_N_2 = sheet_N_Ori
        sheet_P_2 = sheet_P_Ori
        
        Sheet_17_Grade = pd.concat([sheet_N_2, sheet_P_2], axis=0).reset_index(drop=True)
        Sheet_17_Grade = Sheet_17_Grade.drop(["period"], axis=1)
        
        Sheet_17_Grade["考核等級是否為G6"] = Sheet_17_Grade.apply(lambda row: 1 if (row["assessment_grade"]=="G6") else 0, axis = 1)
        Sheet_17_Grade["考核等級是否為G1~G5"] = Sheet_17_Grade.apply(lambda row: 1 if (row["assessment_grade"]!="G6") else 0, axis = 1)

        Sheet_17_is_G6 = Sheet_17_Grade.groupby(["employee_id"], as_index=False)["考核等級是否為G6"].agg({"IS_G6_AGG": np.sum})
        Sheet_17_is_G1G5 = Sheet_17_Grade.groupby(["employee_id"], as_index = False)["考核等級是否為G1~G5"].agg({"IS_G1G5_AGG": np.sum})
        Sheet_17_G6_Cond = pd.merge(Sheet_17_is_G6, Sheet_17_is_G1G5, how="left", on="employee_id")
        Sheet_17_G6_Cond["月考核只有G6"] = Sheet_17_G6_Cond.apply(lambda row: 1 if row["IS_G1G5_AGG"] == 0 and row["IS_G6_AGG"] != 0 else 0, axis=1)
        Sheet_17_G6_Cond = Sheet_17_G6_Cond.drop(["IS_G6_AGG", "IS_G1G5_AGG"], axis=1)
        
        big_sheet = pd.merge(big_sheet, Sheet_17_G6_Cond, how="left", on="employee_id")
        big_sheet["月考核只有G6"] = big_sheet["月考核只有G6"].fillna(0)

        ###########
        # PART 03 # 過去月考核是否有G6
        ###########
        sheet_N_3 = sheet_N_Ori
        sheet_P_3 = sheet_P_Ori
        
        sheet_N_3["月考核是否為G6"] = sheet_N_3.apply(lambda row: 1 if row["assessment_grade"] == "G6" else 0, axis=1)
        sheet_P_3["月考核是否為G6"] = sheet_P_3.apply(lambda row: 1 if row["assessment_grade"] == "G6" else 0, axis=1)

        Sheet_17_Month_Grade_History_N = sheet_N_3.groupby(["employee_id"], as_index=False)["月考核是否為G6"].agg({"共有幾次G6":np.sum})
        Sheet_18_Month_Grade_History_P = sheet_P_3.groupby(["employee_id"], as_index=False)["月考核是否為G6"].agg({"共有幾次G6":np.sum})

        Sheet_17_Month_Grade_History = pd.concat([Sheet_17_Month_Grade_History_N, Sheet_18_Month_Grade_History_P], axis=0).reset_index(drop=True)
        Sheet_17_Month_Grade_History = pd.merge(ID, Sheet_17_Month_Grade_History, how="left", on="employee_id")
        Sheet_17_Month_Grade_History["共有幾次G6"] = Sheet_17_Month_Grade_History["共有幾次G6"].fillna(0)
        Sheet_17_Month_Grade_History["過去月考核是否有G6"] = Sheet_17_Month_Grade_History.apply(lambda row: 1 if row["共有幾次G6"] > 0 else 0, axis=1)
        Sheet_17_Month_Grade_History = Sheet_17_Month_Grade_History.drop(["共有幾次G6"], axis=1)
        
        big_sheet = pd.merge(big_sheet, Sheet_17_Month_Grade_History, how="left", on="employee_id")

        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_2021_preprocess(self, big_sheet, sheet_N_Ori, sheet_P_Ori, ID_Day_N):
        print(f"[資料前處理] Table 20：離職員工於離職前兩年的半年度考核")
        print(f"[資料前處理] Table 21：在職員工近前兩年的半年度考核")
        ###########
        # PART 01 # One-hot Every Assessment
        ###########
        ID = ID_Day_N.drop(["turnover_date"], axis=1)

        ##########
        # 離職人員
        sheet_N = sheet_N_Ori
        
        # 將 G4 丟掉
        Keep_index = sheet_N["assessment_grade"].isin(["G1", "G2", "G3"])
        sheet_N = sheet_N[Keep_index].reset_index(drop=True)
        
        # 計算平均月考核(Only 考慮有 G1~G3 的那群人)
        sheet_N["半年考核"] = sheet_N.apply(lambda row: int(row["assessment_grade"][1]), axis=1)
        AVG_N = sheet_N.groupby(["employee_id"], as_index=False)["半年考核"].agg({"平均半年考核": np.mean})

        # 取得最近月考核(Only 考慮有 G1~G3 的那群人)
        sheet_Reshape_N = sheet_N.groupby(["employee_id"], as_index=False)["period"].agg({"最近期間":np.max})
        sheet_Reshape_N = pd.merge(sheet_Reshape_N, AVG_N, how="left", on="employee_id")
        sheet_Reshape_N = pd.merge(sheet_Reshape_N, sheet_N, how="left", left_on=["employee_id", "最近期間"], right_on=["employee_id", "period"])
        sheet_Reshape_N["最近半年考核"] = sheet_Reshape_N["半年考核"]
        
        # Drop
        sheet_Reshape_N = sheet_Reshape_N.drop(["最近期間", "period", "assessment_grade", "半年考核"], axis=1)
        
        ##########
        # 在職人員
        sheet_P = sheet_P_Ori

        # 將 G6 丟掉
        Keep_index = sheet_P["assessment_grade"].isin(["G1", "G2", "G3"])
        sheet_P = sheet_P[Keep_index].reset_index(drop=True)
        
        # 計算平均月考核(Only 考慮有G1~G5的那群人)
        sheet_P["半年考核"] = sheet_P.apply(lambda row: int(row["assessment_grade"][1]), axis = 1)
        AVG_P = sheet_P.groupby(["employee_id"], as_index = False)["半年考核"].agg({"平均半年考核": np.mean})

        # 取得最近月考核(Only 考慮有G1~G5的那群人)
        sheet_Reshape_P = sheet_P.groupby(["employee_id"], as_index=False)["period"].agg({"最近期間":np.max})
        sheet_Reshape_P = pd.merge(sheet_Reshape_P, AVG_P, how="left", on="employee_id")
        sheet_Reshape_P = pd.merge(sheet_Reshape_P, sheet_P, how="left", left_on=["employee_id", "最近期間"], right_on=["employee_id", "period"])
        sheet_Reshape_P["最近半年考核"] = sheet_Reshape_P["半年考核"]

        # Drop
        sheet_Reshape_P = sheet_Reshape_P.drop(["最近期間", "period", "assessment_grade", "半年考核"], axis=1)

        #############
        # 合併
        sheet = pd.concat([sheet_Reshape_N, sheet_Reshape_P], axis=0).reset_index(drop=True)
        sheet = pd.merge(ID, sheet, how="left", on="employee_id")
        
        # 製作：有無月考核/平均半年考核是否在[1,2)/平均半年考核是否在[2,3]
        sheet["無平均半年考核"] = sheet.apply(lambda row: 1 if np.isnan(row["平均半年考核"]) else 0, axis=1)
        sheet["平均半年考核是否在[1,2)"] = sheet.apply(lambda row: 1 if (row["平均半年考核"] >= 1 and row["平均半年考核"] < 2) else 0, axis=1)
        sheet["平均半年考核是否在[2,3]"] = sheet.apply(lambda row: 1 if (row["平均半年考核"] >= 2 and row["平均半年考核"] <= 3) else 0, axis=1)

        big_sheet = pd.merge(big_sheet, sheet, how="left", on="employee_id")
        ###########
        # PART 02 # 只有 G6
        ###########
        sheet_N_2 = sheet_N_Ori
        sheet_P_2 = sheet_P_Ori
        
        Sheet_19_Grade = pd.concat([sheet_N_2, sheet_P_2], axis=0).reset_index(drop=True)
        Sheet_19_Grade = Sheet_19_Grade.drop(["period"], axis=1)
        
        Sheet_19_Grade["考核等級是否為G4"] = Sheet_19_Grade.apply(lambda row: 1 if (row["assessment_grade"]=="G4") else 0, axis=1)
        Sheet_19_Grade["考核等級是否為G1~G3"] = Sheet_19_Grade.apply(lambda row: 1 if (row["assessment_grade"]!="G4") else 0, axis=1)

        Sheet_19_is_G4 = Sheet_19_Grade.groupby(["employee_id"], as_index=False)["考核等級是否為G4"].agg({"IS_G4_AGG": np.sum})
        Sheet_19_is_G1G3 = Sheet_19_Grade.groupby(["employee_id"], as_index = False)["考核等級是否為G1~G3"].agg({"IS_G1G3_AGG": np.sum})
        Sheet_19_G4_Cond = pd.merge(Sheet_19_is_G4, Sheet_19_is_G1G3, how="left", on="employee_id")
        Sheet_19_G4_Cond["半年考核只有G4"] = Sheet_19_G4_Cond.apply(lambda row: 1 if row["IS_G1G3_AGG"] == 0 and row["IS_G4_AGG"] != 0 else 0, axis=1)
        Sheet_19_G4_Cond = Sheet_19_G4_Cond.drop(["IS_G4_AGG", "IS_G1G3_AGG"], axis=1)
        
        big_sheet = pd.merge(big_sheet, Sheet_19_G4_Cond, how="left", on="employee_id")
        big_sheet["半年考核只有G4"] = big_sheet["半年考核只有G4"].fillna(0)

        ###########
        # PART 03 # 過去月考核是否有G6
        ###########
        sheet_N_3 = sheet_N_Ori
        sheet_P_3 = sheet_P_Ori
        
        sheet_N_3["半年考核是否為G4"] = sheet_N_3.apply(lambda row: 1 if row["assessment_grade"] == "G4" else 0, axis=1)
        sheet_P_3["半年考核是否為G4"] = sheet_P_3.apply(lambda row: 1 if row["assessment_grade"] == "G4" else 0, axis=1)

        Sheet_19_Month_Grade_History_N = sheet_N_3.groupby(["employee_id"], as_index=False)["半年考核是否為G4"].agg({"共有幾次G4":np.sum})
        Sheet_20_Month_Grade_History_P = sheet_P_3.groupby(["employee_id"], as_index=False)["半年考核是否為G4"].agg({"共有幾次G4":np.sum})

        Sheet_19_Month_Grade_History = pd.concat([Sheet_19_Month_Grade_History_N, Sheet_20_Month_Grade_History_P], axis=0).reset_index(drop=True)
        
        Sheet_19_Month_Grade_History = pd.merge(ID, Sheet_19_Month_Grade_History, how="left", on="employee_id")
        Sheet_19_Month_Grade_History["共有幾次G4"] = Sheet_19_Month_Grade_History["共有幾次G4"].fillna(0)
        Sheet_19_Month_Grade_History["過去半年考核是否有G4"] = Sheet_19_Month_Grade_History.apply(lambda row: 1 if row["共有幾次G4"] > 0 else 0, axis=1)
        Sheet_19_Month_Grade_History = Sheet_19_Month_Grade_History.drop(["共有幾次G4"], axis=1)
        
        big_sheet = pd.merge(big_sheet, Sheet_19_Month_Grade_History, how="left", on="employee_id")
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]    
        return big_sheet

    def tbl_22_preprocess(self, big_sheet, sheet, Today):
        print(f"[資料前處理] Table 22：員工調薪次數")

        
        sheet_reshape = sheet.groupby(["employee_id"], as_index=False)["num_of_salary_adjust"].agg({"調薪次數": np.sum})
        big_sheet = pd.merge(big_sheet, sheet_reshape, how="left", on="employee_id")
        big_sheet["調薪次數"] = big_sheet["調薪次數"].fillna(0)
        # NOTICE
        big_sheet["在職時長"] = big_sheet.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["seniority_date"])).days if row["turnover_date"] != "" else (pd.to_datetime(Today) - pd.to_datetime(row["seniority_date"])).days, axis = 1)
        big_sheet["在職時長"] = big_sheet.apply(lambda row: row["在職時長"] if row["在職時長"] < 365*20 else 365*20, axis=1)
        big_sheet["在職時長"] = big_sheet.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)

        big_sheet["平均調薪次數"] = big_sheet.apply(lambda row: row["調薪次數"]/row["在職時長"], axis = 1)
        big_sheet = big_sheet.drop(["在職時長"], axis = 1)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def tbl_23_preprocess(self, big_sheet, sheet_ori, ID_Day_N, Today):
        
        print(f"[資料前處理] Table 23：員工調薪幅度")
        #################################
        # PART 01: 最近調薪幅度是 A/ B/ C
        #################################
        # 先取得最近調薪日資訊
        sheet_ori_reshape = sheet_ori
        sheet_ori_reshape = sheet_ori_reshape.groupby(["employee_id"], as_index = False)["period"].agg({"最近調薪日":np.max})

        # 再透過 Merge 將調薪幅度合併上
        sheet_ori_reshape = pd.merge(sheet_ori_reshape, sheet_ori, how="inner", left_on=["employee_id", "最近調薪日"], right_on=["employee_id", "period"])

        # 將所有的調薪幅度資訊 attach 到 所有員工識別碼上
        sheet_ori_reshape = pd.merge(ID_Day_N["employee_id"], sheet_ori_reshape, how="left", on="employee_id")

        # Drop 掉不需要的資訊
        sheet_ori_reshape = sheet_ori_reshape.drop(["最近調薪日", "period"], axis=1)
        sheet_ori_reshape["range_salary_adjust"] = sheet_ori_reshape["range_salary_adjust"].fillna("D")
        map_val = {"3%":"A", "3to5%": "B", "up10%": "C", "D": "D"}
        sheet_ori_reshape["最近調薪幅度"] = sheet_ori_reshape.apply(lambda row: map_val[row["range_salary_adjust"]], axis = 1)
        sheet_ori_reshape = sheet_ori_reshape.drop(["range_salary_adjust"] ,axis = 1)

        for X in ["A", "B", "C", "D"]:
            sheet_ori_reshape[f"最近調薪幅度是{X}"] = sheet_ori_reshape.apply(lambda row: 1 if (row["最近調薪幅度"]==X) else 0, axis=1)
        sheet_ori_reshape = sheet_ori_reshape.drop(["最近調薪幅度"], axis=1)

        #####################
        # PART 02: 調薪間隔 #
        ####################
        Sheet_22_Adjust_Salary_Range = sheet_ori
        Sheet_22_Adjust_Salary_recent_date = Sheet_22_Adjust_Salary_Range.groupby(["employee_id"], as_index = False)["period"].agg({"最近調薪日":np.max})
        Sheet_22_Adjust_Salary_recent_date = pd.merge(Sheet_22_Adjust_Salary_recent_date, ID_Day_N, how="left", on="employee_id")
        Sheet_22_Adjust_Salary_recent_date["調薪間隔"] = Sheet_22_Adjust_Salary_recent_date.apply(lambda row: (pd.to_datetime(Today) - pd.to_datetime(row["最近調薪日"])).days if row["turnover_date"] == "" else (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["最近調薪日"])).days, axis=1)
        Sheet_22_Adjust_Salary_recent_date = Sheet_22_Adjust_Salary_recent_date.drop(["最近調薪日", "turnover_date"], axis = 1)
        Sheet_22_Adjust_Salary_Range_Reshape = pd.merge(sheet_ori_reshape, Sheet_22_Adjust_Salary_recent_date, how = "left", on = "employee_id")

        
        #################################
        # PART 03: 調薪幅度為 ABC 的次數 #
        #################################
        Sheet_22_Adjust_Salary_Range = sheet_ori
        Sheet_22_Adjust_Salary_Range = Sheet_22_Adjust_Salary_Range.drop(["period"], axis=1)
        map_val = {"3%":"A", "3to5%": "B", "up10%": "C", "D": "D"}
        Sheet_22_Adjust_Salary_Range["調薪幅度ABC"] = Sheet_22_Adjust_Salary_Range.apply(lambda row: map_val[row["range_salary_adjust"]], axis=1)
        Sheet_22_Adjust_Salary_Range = Sheet_22_Adjust_Salary_Range.drop(["range_salary_adjust"], axis=1)
        
        Sheet_22_Adjust_Salary_Range["調薪幅度為A"] = Sheet_22_Adjust_Salary_Range.apply(lambda row: 1 if row["調薪幅度ABC"] == "A" else 0, axis = 1)
        Sheet_22_A = Sheet_22_Adjust_Salary_Range.groupby(["employee_id"], as_index=False)["調薪幅度為A"].agg({"調薪幅度為A的累積次數": np.sum})

        Sheet_22_Adjust_Salary_Range["調薪幅度為B"] = Sheet_22_Adjust_Salary_Range.apply(lambda row: 1 if row["調薪幅度ABC"] == "B" else 0, axis = 1)
        Sheet_22_B = Sheet_22_Adjust_Salary_Range.groupby(["employee_id"], as_index=False)["調薪幅度為B"].agg({"調薪幅度為B的累積次數": np.sum})
        
        Sheet_22_Adjust_Salary_Range["調薪幅度為C"] = Sheet_22_Adjust_Salary_Range.apply(lambda row: 1 if row["調薪幅度ABC"] == "C" else 0, axis = 1)
        Sheet_22_C = Sheet_22_Adjust_Salary_Range.groupby(["employee_id"], as_index=False)["調薪幅度為C"].agg({"調薪幅度為C的累積次數": np.sum})

        Sheet_22_Adjust_Salary_Range_Reshape = pd.merge(Sheet_22_Adjust_Salary_Range_Reshape, Sheet_22_A, how="left", on="employee_id")
        Sheet_22_Adjust_Salary_Range_Reshape = pd.merge(Sheet_22_Adjust_Salary_Range_Reshape, Sheet_22_B, how="left", on="employee_id")
        Sheet_22_Adjust_Salary_Range_Reshape = pd.merge(Sheet_22_Adjust_Salary_Range_Reshape, Sheet_22_C, how="left", on="employee_id")

        big_sheet = pd.merge(big_sheet, Sheet_22_Adjust_Salary_Range_Reshape, how = "left", on = "employee_id")

        # big_sheet 衍生計算
        big_sheet["調薪幅度為A的累積次數"] = big_sheet["調薪幅度為A的累積次數"].fillna(0)
        big_sheet["調薪幅度為B的累積次數"] = big_sheet["調薪幅度為B的累積次數"].fillna(0)
        big_sheet["調薪幅度為C的累積次數"] = big_sheet["調薪幅度為C的累積次數"].fillna(0)
        
        big_sheet["離職日+今天"] = big_sheet.apply(lambda row: row["turnover_date"] if row["turnover_date"] != "" else Today, axis=1)
        big_sheet["調薪間隔"] = big_sheet.apply(lambda row: (pd.to_datetime(row["離職日+今天"]) - pd.to_datetime(row["seniority_date"])).days if np.isnan(row["調薪間隔"]) else row["調薪間隔"], axis=1)

        big_sheet["在職時長"] = big_sheet.apply(lambda row: (pd.to_datetime(row["離職日+今天"]) - pd.to_datetime(row["seniority_date"])).days, axis=1)
        big_sheet["在職時長"] = big_sheet.apply(lambda row: row["在職時長"] if row["在職時長"] < 365*20 else 365*20, axis=1)
        big_sheet["在職時長"] = big_sheet.apply(lambda row: row["在職時長"] if row["在職時長"] != 0 else 1, axis=1)

        big_sheet["平均調薪幅度為A的次數"] = big_sheet.apply(lambda row: row["調薪幅度為A的累積次數"]/row["在職時長"], axis=1)
        big_sheet["平均調薪幅度為B的次數"] = big_sheet.apply(lambda row: row["調薪幅度為B的累積次數"]/row["在職時長"], axis=1)
        big_sheet["平均調薪幅度為C的次數"] = big_sheet.apply(lambda row: row["調薪幅度為C的累積次數"]/row["在職時長"], axis=1)
        big_sheet = big_sheet.drop(["離職日+今天", "在職時長"], axis = 1)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def tbl_24_preprocess(self, big_sheet, sheet_ori):
        print(f"[資料前處理] Table 24：員工水位")

        Sheet_23_Water_Level = sheet_ori.groupby(["employee_id"], as_index = False)["interval_value"].agg({"區間值":np.mean})
        Sheet_23_Water_Level["薪資水位"] = Sheet_23_Water_Level["區間值"]
        Sheet_23_Water_Level = Sheet_23_Water_Level.drop(["區間值"], axis=1)
        big_sheet = pd.merge(big_sheet, Sheet_23_Water_Level, how ="left", on = "employee_id")
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def operation_25_preprocess(self, big_sheet, Today):
        big_sheet["年紀(用離職日+今天扣)"]  = big_sheet.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["birthday"])).days if (row["turnover_date"] != "") else (pd.to_datetime(Today) - pd.to_datetime(row["birthday"])).days, axis = 1)
        big_sheet["年資(用離職日+今天扣)"]  = big_sheet.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["seniority_date"])).days if (row["turnover_date"] != "") else (pd.to_datetime(Today) - pd.to_datetime(row["seniority_date"])).days, axis = 1)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def operation_26_preprocess(self, big_sheet, Today):
        big_sheet["離職日+今天"] = big_sheet.apply(lambda row: row["turnover_date"] if row["turnover_date"] != "" else Today, axis=1)
        big_sheet["升等間隔"] = big_sheet.apply(lambda row: (pd.to_datetime(row["turnover_date"]) - pd.to_datetime(row["upgrade_grade_date"])).days/365 if row["turnover_date"] != "" else (pd.to_datetime(Today) - pd.to_datetime(row["upgrade_grade_date"])).days/365, axis = 1)
        big_sheet["升職間隔"] = big_sheet.apply(lambda row: (pd.to_datetime(row["離職日+今天"]) - pd.to_datetime(row["promote_date"])).days/365 if (row["promote_date"] != "") else (pd.to_datetime(row["離職日+今天"]) - pd.to_datetime(row["seniority_date"])).days/365, axis = 1)
        big_sheet = big_sheet.drop(["離職日+今天"], axis = 1)

        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def operation_27_preprocess(self, big_sheet):
        big_sheet["最近停年"] = big_sheet.apply(lambda row: row["升等間隔"] if (np.isnan(row["最近停年"]) == True) else row["最近停年"], axis = 1)
        big_sheet["平均停年"] = big_sheet.apply(lambda row: row["升等間隔"] if (np.isnan(row["平均停年"]) == True) else row["平均停年"], axis = 1)
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def operation_28_preprocess(self, big_sheet):

        big_sheet["有無升等"] = big_sheet.apply(lambda row: True if (pd.to_datetime(row["upgrade_grade_date"]) > pd.to_datetime(row["seniority_date"])) else False, axis = 1)
        big_sheet["有無升職"] = big_sheet["promote_date"] != ""

        big_sheet = big_sheet.drop(["調薪次數", "調薪幅度為A的累積次數", "調薪幅度為B的累積次數", "調薪幅度為C的累積次數"], axis = 1)
            
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def operation_29_preprocess(self, big_sheet):

        null_index = big_sheet["平均訓練次數"].notnull()
        big_sheet = big_sheet[null_index]
        null_index = big_sheet["薪資水位"].notnull()
        big_sheet = big_sheet[null_index]
        
        # Drop possibly duplicated rows
        unduplicate_index =  big_sheet["employee_id"].duplicated() == False
        big_sheet = big_sheet[unduplicate_index]
        return big_sheet

    def operation_30_preprocess(self, big_sheet):
        
        col_dict = {
            'employee_id': '員工識別碼', #0
            'direct_and_indirect': '直間接', #1
            'company_code': '公司代碼', #2
            'division_code': '廠處代碼', #3
            'department_code': '部門代碼', #4
            'group_code': '組站代碼', #5
            'division': '廠別', #6
            'employment_class_code': '雇用類別代碼', #7
            'job_code': '職務代碼', #8
            'job_grade_code': '職等代碼', #9
            'job_rank_code': '職級代碼', #10
            'job_class_code': '職類代碼', #11
            'class_type': '班別', #12
            'punch_if': '是否須打卡', #13
            'job_state': '在職狀態', #14
            'gender': '性別', #15
            'birthday': '生日', #16
            'household_county_code': '戶籍地縣市代碼', #17
            'household_zipcode_code': '戶籍地區域代碼', #18
            'communication_county_code': '通訊地縣市代碼', #19
            'communication_zipcode_code': '通訊地區域代碼', #20
            'education_code': '學歷代碼', #21
            'faculty_code': '科系代碼', #22
            'marital_status': '婚姻狀況', #23
            'seniority_date': '年資日', #24
            'upgrade_grade_date': '升等日', #25
            'promote_date': '升職日', #26
            'demotion_date': '降職日', #27
            'tlws_date': '留職停薪日', #28
            'reinstatement_date': '復職日', #29
            'turnover_date': '離職日' #30
            
        }
        
        big_sheet.rename(columns=col_dict, inplace=True)
        return big_sheet
