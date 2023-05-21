import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="xxxxxxxx" # Enter The Password of your localhost
)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE predicteddata")

mycursor.execute("USE predicteddata")

mycursor.execute("CREATE TABLE predicted (id INT AUTO_INCREMENT PRIMARY KEY, model_name VARCHAR(255), class_name VARCHAR(255), class_prob FLOAT)")

mydb.close()

