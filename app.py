from flask import Flask
app = Flask("example")
import listener

if __name__ == '__main__':
    app.debug = True
    app.run()