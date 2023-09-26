import sqlalchemy
import pymysql
import sys
from sqlalchemy import text
import data_storage
from datetime import datetime

pymysql.install_as_MySQLdb()
bridge = True
while bridge:
    operation = input('Please indicate which type of operation you want to implement (Choose from add, delete, display):\n')
    portfolio = []
    time = []
    stock = []
    engine = sqlalchemy.create_engine("mysql+mysqldb://root:Dsci-560@localhost")	    
    with engine.connect() as conn:
        dateResult = conn.execute(text(f"SELECT table_name, create_time FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = 'yahoo_finance' AND table_name like '%content'")).fetchall()
        for row in dateResult:
            portfolio.append(row[0].split('_')[0])
            time.append(row[1].strftime("%m/%d/%Y, %H:%M:%S"))
        if operation == 'display':
            print('Creation time of all portfolios: ')
            print(list(zip(portfolio, time)))
        db = input('Please enter the name of the portfolio you want to explore/edit: ')
        if db not in portfolio:
            print('Wrong portfolio name! Please try again.')
            continue
        stockResult = conn.execute(text(f"select column_name from INFORMATION_SCHEMA.COLUMNS where table_name = '{db}_history'")).fetchall()
        print('Current stocks inside this portfolio: ')
        for row in stockResult:
            if row[0] != 'Date':
                print(row[0])
                stock.append(row[0])
       # when operation is add or delete
        if operation == 'add':
            addStock = input("Please type in the symbol of stock which you want to add in this portfolio: ")
            weightInput = input("Please type in the weight for renewed each stock (Use comma and space to separate. Sample input: 0.3, 0.2, 0.2, 0.1, 0.2):\n")
            weight = weightInput.split(", ")
            weight = [float(i) for i in weight]
            stock.append(addStock)
            data_storage.storage(stock, weight, db)
        elif operation == 'delete':
            delStock = input("Please type in the symbol of stock which you want to delete in this portfolio: ")
            weightInput = input("Please type in the weight for renewed each stock (Use comma and space to separate. Sample input: 0.3, 0.2, 0.2, 0.1, 0.2):\n")
            if delStock not in stock:
                print('Wrong symbol! Please try again.')
            weight = weightInput.split(", ")
            weight = [float(i) for i in weight]
            stock.remove(delStock)
            data_storage.storage(stock, weight, db)
    survey = input('Still want to continue?(y/n)')
    if survey == 'y':
        bridge = True
    else:
        bridge = False
	    
