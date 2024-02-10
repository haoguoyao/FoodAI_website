from app import app

from flask import Flask


app.run(host="0.0.0.0",port = 80,debug = True,threaded=True)
# app.run(host="0.0.0.0",port = 5000,debug = True)
