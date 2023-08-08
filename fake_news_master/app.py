from flask import Flask, escape, request, render_template
import pred


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        head = str(request.form['news'])
        title = str(request.form['texta'])
        print(head)

        probality,predict = pred.fake_prediction(head,title)
        
        if predict == 1:
            out = 'real'
        else:
            out ='misleading'

        return render_template("prediction.html", prediction_text=f"News article can be said to be  -> {out} with a confidence of {probality}")


    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()

