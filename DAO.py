import sqlite3

conn = sqlite3.connect('food.db')
print ("Opened database successfully")
c = conn.cursor()
# create table if it is not exist
try:
    c.execute('''CREATE TABLE FOOD
        (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
        food_type           TEXT    NOT NULL,
        ingredients           TEXT     NOT NULL,
        image_name        TEXT,
        noise   TEXT,
        histogram_equalization  INT,
        feedback        INT,
        geo_location          TEXT    NOT NULL,
        browser_name    TEXT    NOT NULL);''')
    print ("Table created successfully")
except:
    print("Table already exist")
conn.commit()


def rate_image(image_name,feedback):
    conn = sqlite3.connect('food.db')
    c = conn.cursor()
    statement = "UPDATE FOOD SET feedback = {} WHERE  image_name= '{}';".format(feedback,image_name)
    c.execute(statement)
    conn.commit()

def generate_a_image(food_type,ingredients,image_name,geo_location,browser_name,histogram_equalization,noise):
    conn = sqlite3.connect('food.db')
    c = conn.cursor()
    statement = "INSERT INTO FOOD (food_type,ingredients,image_name,geo_location,browser_name,histogram_equalization,noise) values ('{}','{}','{}','{}','{}','{}','{}');".format(food_type,ingredients,image_name,geo_location,browser_name,histogram_equalization,noise)
    c.execute(statement)
    conn.commit()

def query_database(sql):
    conn = sqlite3.connect('food.db')
    c = conn.cursor()
    statement = sql
    result = c.execute(statement)
    return result


# query_result = query_database("select * from food")
# for i in query_result:
#     print(i)
