import pandas as pd
import pypyodbc

class DB_process():

    def __init__(self):
        self.SERVER_NAME = "DESKTOP-ITN4K40\SQLEXPRESS"
        self.DB_NAME = "ASE_007"
        self.CONN = pypyodbc.connect("""
            Driver={{SQL Server Native Client 11.0}};
            Server={0};
            Database={1};
            Trusted_Connection=yes;""".format(self.SERVER_NAME, self.DB_NAME)
        )
        self.TBL_NAME_MAP = {
            "TBL01": "Employee_Sheet", # 員工資料
            "TBL02": "LeaveEmployee_AbsenceSheet_Prevoius_1year",     # 離職員工於前一年的請假單
            "TBL03": "InServiceEmployee_AbsenceSheet_Prevoius_1year", # 在職員工於前一年的請假單
            "TBL04": "Leave_Employee_TrainingTime_Prevoius_2year",     # 離職人員於離職前兩年的在職訓練次數 
            "TBL05": "InService_Employee_TrainingTime_Prevoius_2year", # 在職人員於離職前兩年的在職訓練次數
            "TBL06": "Leave_Employee_OverTime_Prevoius_1year",     # 離職員工於離職前一年的加班時數
            "TBL07": "InService_Employee_OverTime_Prevoius_1year", # 在職員工於離職前一年的加班時數
            "TBL08": "Leave_Employee_Preemployment_exp",     # 離職員工職前經驗年資
            "TBL09": "InService_Employee_Preemployment_exp", # 在職員工職前經驗年資
            "TBL10": "Leave_Employee_Preemployment_Num_Of_Company",     # 離職員工職前經驗公司數
            "TBL11": "InService_Employee_Preemployment_Num_Of_Company", # 在職員工職前經驗公司數
            "TBL12": "Leave_Employee_Num_Of_Upgrade",     # 離職員工升等次數
            "TBL13": "InService_Employee_Num_Of_Upgrade", # 在職員工升等次數
            "TBL14": "Leave_Employee_Num_Of_Downgrade",     # 離職員工降等次數
            "TBL15": "InService_Employee_Num_Of_Downgrade", # 在職員工降等次數
            "TBL16": "Leave_Employee_Job_Grade_Suspend_Year",     # 離職員工職等停年
            "TBL17": "InService_Employee_Job_Grade_Suspend_Year", # 在職員工職等停年
            "TBL18": "Leave_Employee_Pre2year_MonthAssessment",     # 離職員工於離職前兩年的月考核
            "TBL19": "InService_Employee_Pre2year_MonthAssessment", # 在職員工近前兩年的月考核
            "TBL20": "Leave_Employee_Pre2year_HalfYearAssessment",     # 離職員工於離職前兩年的半年度考核
            "TBL21": "InService_Employee_Pre2year_HalfYearAssessment", # 在職員工於前兩年的半年度考核
            "TBL22": "Num_Of_Employee_Salary_Adjust", # 員工調薪次數
            "TBL23": "Range_Of_Employee_Salary_Adjust", # 員工調薪幅度
            "TBL24": "Interval_Of_Employee_Salary", # 員工水位
        }

    def connect_db_and_query(self, sql_query, tbl_name, idx):
        df = pd.read_sql(sql_query, self.CONN)
        print(f"[{self.DB_NAME}] 讀取 DB Table {idx+1}：{tbl_name}，共獲取 {len(df)} 筆資料。")
        return df

    def select_db_data(self):
        TBL_KEYS = list(self.TBL_NAME_MAP.keys())
        BIG_SHEETS = list()
        for idx, TBL_Key in enumerate(TBL_KEYS):
            TBL_NAME = self.TBL_NAME_MAP[TBL_Key]
            SQL_QUERY = f"SELECT * FROM {TBL_NAME}"
            sub_sheet = self.connect_db_and_query(sql_query=SQL_QUERY, tbl_name=TBL_NAME, idx=idx)
            BIG_SHEETS.append(sub_sheet)
        return BIG_SHEETS

    def update_module_status2(self, Block_Num):

        """
        模式 status 有2個變化情況
        0->1 時機點: 開始儲存訓練資料
        1->2 時機點: 完成儲存訓練資料

        """

        if Block_Num == 1:
            Cursor=self.CONN.cursor()
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 1 WHERE code = 1")
            self.CONN.commit()

        elif Block_Num == 2:
            Cursor=self.CONN.cursor()
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 2 WHERE code = 1")
            self.CONN.commit()



    def update_module_status(self, Block_Num, thresh_prob_XGB=0, thresh_prob_NN=0):

        """
        模式 status 有七個變化情況
        0->1 時機點: 開始生成訓練大表 & 預測大表
        1->2 時機點: 開始訓練XGboost
        2->3 時機點: 用XGboost預測機率
        3->4 時機點: 開始訓練NN
        4->5 時機點: 用NN預測機率
        5->6 時機點: 將預測機率塞回DB
        6->0 DBA檢查
        """

        if Block_Num == 1:
            Cursor=self.CONN.cursor()
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 1 WHERE code = 2")
            self.CONN.commit()

        elif Block_Num == 2:
            Cursor=self.CONN.cursor()
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 2 WHERE code = 2")
            self.CONN.commit()

        elif Block_Num == 3:
            Cursor=self.CONN.cursor()
            thresh_prob_str = str(thresh_prob_XGB) + ',' + str(thresh_prob_NN)
            print('str:', thresh_prob_str)
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 3,Date = '" + thresh_prob_str + "' WHERE code = 2")

            self.CONN.commit()

        elif Block_Num == 4:
            Cursor=self.CONN.cursor()
            thresh_prob_str = str(thresh_prob_XGB) + ',' + str(thresh_prob_NN)
            print('str:', thresh_prob_str)
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 4,Date = '" + thresh_prob_str + "' WHERE code = 2")
            self.CONN.commit()

        elif Block_Num == 5:
            Cursor=self.CONN.cursor()
            thresh_prob_str = str(thresh_prob_XGB) + ',' + str(thresh_prob_NN)
            print('str:', thresh_prob_str)
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 5,Date = '" + thresh_prob_str + "' WHERE code = 2")

            self.CONN.commit()

        elif Block_Num == 6:
            Cursor=self.CONN.cursor()
            thresh_prob_str = str(thresh_prob_XGB) + ',' + str(thresh_prob_NN)
            print('str:', thresh_prob_str)
            Cursor.execute("UPDATE ASE_007.dbo.Module_Status SET status = 6,Date = '" + thresh_prob_str + "' WHERE code = 2")
            self.CONN.commit()

    def insert_result(self, RESULT_PATH):
        df = pd.read_csv(RESULT_PATH)
        for row in df.values:
            Cursor=self.CONN.cursor()
            target = "INSERT INTO ASE_007.dbo.ModelPerdictData ([Employee_Id],[Generate_Date],[Turnover_Rate],[Department_Code],[Module]) VALUES ('"+str(row[0])+"','"+str(row[1])+"','"+str(row[2])+"','"+str(row[3])+"','"+str(row[4])+"')"
            Cursor.execute(target)
            self.CONN.commit()