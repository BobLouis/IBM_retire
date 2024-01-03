import sys
import os
import pypyodbc
import time
import schedule

SERVER_NAME = "DESKTOP-ITN4K40\SQLEXPRESS"
DB_NAME = "ASE_007"

CONN = pypyodbc.connect("""
    Driver={{SQL Server Native Client 11.0}};
    Server={0};
    Database={1};
    Trusted_Connection=yes;""".format(SERVER_NAME, DB_NAME)
)

def job():
    cursor = CONN.cursor()
    # 儲存訓練資料
    cursor.execute("SELECT * FROM ASE_007.dbo.Module_Status WHERE code = 1")
    rows = cursor.fetchall()
    for row in rows:
        if row[1] == "1":
            os.system("python src.py SAVE "+row[2]+" ")

    # 訓練並預測
    cursor.execute("SELECT * FROM ASE_007.dbo.Module_Status WHERE code = 2")
    rows2 = cursor.fetchall()
    for row2 in rows2:
        if row2[1] == "1":
            os.system("python src.py PREDICT "+row2[2]+" ")
            

schedule.every(5).seconds.do(job) #每5秒執行job

  
while True:  
    schedule.run_pending()
    time.sleep(1)