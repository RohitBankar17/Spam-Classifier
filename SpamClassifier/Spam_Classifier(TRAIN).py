import numpy as np
import csv,math,json,math

train_X_file_path="train_X_nb.csv"
train_Y_file_path="train_Y_nb.csv"

def import_data(train_X_file_path,train_Y_file_path):
    train_X=np.genfromtxt(train_X_file_path,delimiter='\n',dtype=str)
    train_Y=np.genfromtxt(train_Y_file_path,delimiter='\n',dtype=str)
    return train_X,train_Y


def preprocessing(X):
    result = []
    for s in X:
        S = s
        for i in range(len(S)):
            if not(S[i].isalpha()) and not(S[i] == " "):
                s = s.replace(S[i], "")
        new = s.strip()
        new = new.lower()
        while not(new.find("  ") == -1):
            i = new.find("  ")
            new = new[0:i] + new[i+1::]
        result.append(new)
    return result

def write_model(model,path):
    a=json.dumps(model)
    with open(path,"w") as outfile:
        outfile.write(a)
        
def class_wise_words_frequency_dict(X, Y):
    class_wise_frequency_dict=dict()
    
    for i in range(len(X)):
        words=X[i].split()
        y=Y[i]
        if y not in class_wise_frequency_dict:
            class_wise_frequency_dict[y]=dict()
        for w in words:
            if w not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[y][w]=0
            class_wise_frequency_dict[y][w]+=1
    
    return class_wise_frequency_dict


def get_class_wise_denominators_likelihood(X, Y,mod):
    vocab=[]
    freq_dict=class_wise_words_frequency_dict(X,Y)
    class_wise_denominators=dict()
    
    for c in mod.keys():
        temp=freq_dict[c]
        class_wise_denominators[c]=sum(list(temp.values()))
        vocab+=list(temp.keys())
    
    vocab=list(set(vocab))
    
    for c in mod.keys():
        class_wise_denominators[c]+=len(vocab)
        
    return class_wise_denominators


def compute_prior_probabilities(Y):
    classes = list(set(Y))
    num= len(Y)
    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = Y.count(c) / num
    return prior_probabilities 


def train(X_p,y_p):
    train_X,train_Y=import_data(X_p,y_p)
    train_X=preprocessing(train_X)
    model=[list(set(train_Y)),class_wise_words_frequency_dict(train_X,train_Y)]
    model.append(get_class_wise_denominators_likelihood(train_X,train_Y,model[1]))
    model.append(compute_prior_probabilities(list(train_Y)))
    write_model(model,"MODEL_FILE.json")
    
    
train(train_X_file_path,train_Y_file_path)