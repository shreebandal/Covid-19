from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=['GET','POST'])
def shree(): 
    if request.method == "POST":
        mydict = request.form
        fever = int(mydict['fever'])
        dryCough = int(mydict['cough'])
        fatigue = int(mydict['fatigue'])
        sputum = int(mydict['sputum'])
        breath = int(mydict['breath'])
        arthralgia = int(mydict['arthralgia'])
        soreThroat = int(mydict['throat'])
        headache = int(mydict['headache'])
        chills = int(mydict['chills'])
        vomiting = int(mydict['vomiting'])
        nasal = int(mydict['nasal'])
        age = int(mydict['age'])

        infprob = clf.predict_proba([[fever,dryCough,fatigue,sputum,breath,arthralgia,soreThroat,headache,chills,vomiting,nasal,age]])[0][1]
        print(infprob)

        return render_template('show.html',inf=round(infprob*100))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)