import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import seaborn as sns
import numpy as np

X= pd.read_csv("date_finale.csv",sep=";",decimal=",")

# Setul de date este bazată pe mediul tehnologic , unde se pune accent pe legatura dintre cumpărător
#forta de muncă și nivelul de educație în ceea ce priveste tehnologia

# fiecare indice X are semnificația sa
# . x2 - PIB pe locuitor
# . x3- Utilizarea   internetului/viteză de către pers.
# . x4 - Pers. care au comandat bunuri și servicii de pe net
# . x5 - Rata angajare în dom. tehnologiei
# . x6 - Demografia afacerilor și întrep. cu crestere mare
# . x7 - Salarii angajați
# . x9 - Rata de angajare a tinerilor neaflați în educatie
# . x10 - Resurse umane în știință

#Verificarea datelor
print(X.dtypes)

# 1. Definirea listei de indicatori economici
indicatori = ["x2", "x3", "x4", "x5", "x6", "x7", "x9", "x10"]

# 3. Eliminăm rândurile cu valori lipsă
X.dropna(inplace=True)
print(X)

# 4. Gruparea și agregarea datelor
X_grouped = X.groupby("GEO (Codes)")[["x2", "x3", "x4", "x5", "x6", "x7", "x10"]].mean()
print(X_grouped)

# 5. Crearea unei liste și a unui dicționar cu regiunile și angajarea în tehnologie
regiuni = X["GEO (Codes)"].tolist()
angajare_tehnologie = X["x5"].tolist()
date_dict = dict(zip(regiuni, angajare_tehnologie))
print(date_dict)

# 6. Folosirea tuplurilor și seturilor pentru a stoca date unice
regiuni_unice = set(regiuni)  # Eliminăm duplicatele

# 7.filtrarea datelor in functie cu loc si iloc
print(X.loc[X["GEO (Codes)"] == "NL32"])  # Am filtrat datele pentru NL32
print(X.iloc[:5, :])  # Primele 5 rânduri

# 8.apelarea unei functii
def analiza_regiune(regiune):
    """Afișează rata de angajare în tehnologie pentru o regiune"""
    if regiune in date_dict:
        print(f"Rata de angajare în tehnologie pentru {regiune} este {date_dict[regiune]}%")
    else:
        print("Regiunea nu există în setul de date.")

analiza_regiune("NL32")

# 9. Date descriptive
print(X.describe(include='all'))
X_categorii=X.select_dtypes(include='object')
print(X_categorii.describe())
print(X.describe(include='object'))
print(X["x5"].describe())
#Media ratei de angajare este în jur de 4%
print(X["x2"].describe())
#Media PIB este de 30000
print(X["x9"].describe())

# 10. Îmbinăm aceste subseturi pe baza coloanei "Regiune" cu comanda left
subset_economic = X[["GEO (Codes)", "x2", "x7"]]  # PIB și salarii
subset_angajare = X[["GEO (Codes)", "x5", "x9"]]  # Rata de angajare și angajare tineri
X_combined = pd.merge(subset_economic, subset_angajare, on="GEO (Codes)", how="left")
print(X_combined.head())

# 11. Reprezentare grafică a distribuției angajării în tehnologie
X["x5"].plot(kind="bar")
plt.xlabel("GEO (Codes")
plt.ylabel("Rata de angajare in tehnologie")
plt.show()

# 12. Histograma pentru rata somajului
print(X["x5"])
X["x5"].plot(kind='hist')
plt.xlabel("Rata somajului")
plt.show()
#putem observa ca cea mai mare rata a somajului este între 4 si 5

# 13. PIB-ul total pentru toate regiunile în parte ( suma pentru fiecare initială)
plot_data = X.copy()
plot_data["Initiala"] = plot_data["GEO (Codes)"].str[0]
plot_data = plot_data.groupby("Initiala")["x2"].sum()
plot_data.sort_values().plot(kind="bar")
plt.xlabel("Inițială Regiune")
plt.ylabel("Suma PIB")
plt.title("Suma PIB pentru fiecare inițială de regiune ")
plt.show()
# Putem observa ca PIB-ul cel mai mare îl are zona I , urmat de N și F
# De asemenea cele mai mici sunt L,B,P,R,A, și H care au sub 100000

# 14. Am realizat un bar char care să prezinte PIB-ul total pentru toate regiunile cu aceeași initiala
# Dar care au rata de angajare în domeniul tehnologic de peste 4
plot_data = X[(X["x5"] >= 4)].copy()
plot_data["Initiala"] = plot_data["GEO (Codes)"].str[0]
plot_data = plot_data.groupby("Initiala")["x2"].sum()
plot_data.sort_values().plot(kind="bar")
plt.xlabel("Inițială Regiune")
plt.ylabel("Suma PIB")
plt.title("Suma PIB pentru fiecare inițială de regiune (4 ≤ x5 ≤ 5)")
plt.show()

# Se observa faptul că regiunile care încep cu B , P , R , H si E PIB-ul total este foarte mic
# Ceea ce poate însemna *1. Situatia economică din aceste regiuni nu este una foarte bună
#                       *2. Rata de angajare pentru dom. tehnologic din aceste regiuni este una foarte mică , mult regiuni având sub 4%
# De asemenea putem observa că tările nordice au rata cea mai mare de angajare în domeniul tehnologic
# Un alt lucru foarte important este faptul că zonele cu initialele L și A au 0 regiuni cu rata de angajare peste 4% iar
# zona I care are PIB-ul total cel mai mare , a ajuns pe locul 4 în ceea ce privește rata de angajare peste 4% , semn că
# regiunile S , F și N au un domeniu dezvoltat în tehnologie mult mai bun , cu o rată de angajare mult mai mare

# 15. Returnează primele n regiuni cu cea mai mare rată de angajare în tehnologie
def top_regiuni_tehnologie(n=20):

    top = X.nlargest(n, "x5")
    return top[["GEO (Codes)", "x5"]]

print(top_regiuni_tehnologie(20))

# 16. Folosirea unei bucle pentru a verifica dacă există regiuni cu PIB sub o anumită valoare
def regiuni_pib(valuare):
    for index, row in X.iterrows():
        if row["x2"] < valuare:
            print(f"{row["GEO (Codes)"]} are un PIB pe locuitor de {row["x2"]}")

regiuni_pib(30000)

# Putem observa că regiunile nordice au PIB-ul peste medie .

# 17. Funcția pentru a afișa regiunile cu rata șomajului sub 25%
def regiuni_angajare():

    somaj = X[X["x9"] < 80]
    return somaj[["GEO (Codes)", "x9"]]

print(regiuni_angajare())

#Putem observa că regiunile cu initialele I și E sunt cele mai multe în ceea ce priveste rata mica de angajare a personelor fără educatie
# semn că în aceste regiuni se pune accent mai mult pe educație
#De asemenea observăm ca zonele cu initiale cum ar fi B , A , L , R ,C care au 0 sau extrem de putine regiuni, unde desi rata de angajare este peste 80%
# ,  lucru care arata că se fac multe angajari si fără educiatie ( exceptand domeniul tehnologic ) , tot au un PIB extrem de mic , semn că zonele respective
# au un nivel extrem de slab economic
if 'x2' in X.columns and 'x5' in X.columns:
  plt.figure(figsize=(8, 6))
  plt.scatter(X['x2'], X['x5'], alpha=0.3, s=10, color='navy')
  plt.title("Analiza regiunilor dezvoltate")
  plt.xlabel("PIB pe locuitori")
  plt.ylabel("Angajare in tehnologie")
  plt.tight_layout()
  plt.show()
#Am prezentat regiunile pe o harta in functie de rata de angajare in tehnologie si PIB , cu cat punctele
#sunt mai indepartate in dreapta si sus , cu atat sunt mai dezvoltate

sns.boxplot(data=X, y="x2")
plt.show()
#Putem observa ca avem doi outlieri in graficul boxplot pentru PIB , care reprezinta valori mult peste PIBurile normale
Q1 = X["x2"].quantile(0.25)
Q3 = X["x2"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outlier_regiuni = X[(X["x2"] < lower_limit) | (X["x2"] > upper_limit)]
print(outlier_regiuni[["GEO (Codes)", "x2"]])
#Observam ca regiunile cu PIB-ul extrem de mare sunt DK01 si SE11
X["log_x2"] = np.log(X["x2"])
#Am logaritmat ca sa scapam de outlieri

sns.boxplot(data=X, y="x5")
plt.show()
Q4 = X["x5"].quantile(0.25)
Q6 = X["x5"].quantile(0.75)
IQR = Q6 - Q4
lower_limit_1 = Q4 - 1.5 * IQR
upper_limit_2 = Q6 + 1.5 * IQR
outlier_regiuni_tehn = X[(X["x5"] < lower_limit_1) | (X["x5"] > upper_limit_2)]
print(outlier_regiuni_tehn[["GEO (Codes)", "x5"]])
X["log_x5"] = np.log(X["x5"])
#Acelasi lucru am facut si pentru rata de angajare in domeniul tehnologic

X_cluster = X[["x2", "x5"]]
# Creăm modelul KMeans cu 3 clustere
kmeans = KMeans(n_clusters=3, random_state=0)
X["cluster"] = kmeans.fit_predict(X_cluster)
# Vizualizăm clusterele
plt.figure(figsize=(8, 6))
plt.scatter(X["x2"], X["x5"], c=X["cluster"], cmap="viridis", s=50)
plt.xlabel("PIB (x2)")
plt.ylabel("Rata angajare în tehnologie (x5)")
plt.title("Clusterizare KMeans")
plt.show()
#Am impartit clusterul in 3 pentru a impartii PIB-ul in: tari sarace , tari mijlocii , tari bogate
#Observa ca exista oarecum o diferenta , insa nu foarte mare la rata de angajare , nefiing un factor atat de importat
#Insa putem observa ca tarile foarte bogate , au o rata foarte mare de angajare in domeniul tehnologic
#Ceea ce indica faptul ca pentru acestea , domeniul tehnologic este unul foarte important


X["high_tech"] = (X["x5"] >= 4).astype(int)

# Alegem predictori (exemplu: PIB și resurse umane în știință)
features = ["x2", "x10"]
X_train, X_test, y_train, y_test = train_test_split(X[features], X["high_tech"], test_size=0.2, random_state=42)
# Modelul logistic
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Predicții și evaluare
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
#La finalul predictiei , observam faptul ca modelul a avut un scor de 64% pentru prezicerea atat a regiunilor
# cu x5 mai are de 4 (1) si cele mai mici decat 4 (0). Este decent dar in continuare o sa mai facem si alte teste
X_reg = X[["x3", "x4", "x6"]]
X_reg = sm.add_constant(X_reg)
# Variabila dependentă
y = X["x2"]
# Modelul
model = sm.OLS(y, X_reg).fit()
# Rezultate
print(model.summary())
#modelul explică 50.2% din variația PIB-ului pe locuitor. E decent pentru date socio-economice.
#ajustează R² în funcție de numărul de variabile. Aproape de R² → înseamnă că toate variabilele contribuie rezonabil.
#x3 (utilizarea internetului)	685.89	p = 0.078 ️	Nu este semnificativ statistic la p < 0.05, dar aproape. Posibilă influență pozitivă asupra PIB
#x4 (comenzi online)	463.35	p = 0.004 	Semnificativ. O creștere cu 1 unitate a acestui indicator → crește PIB-ul cu ~463 unități
#x6 (afaceri/întreprinderi în creștere)	-1005.39	p = 0.005 	Semnificativ. Surprinzător, dar coeficient negativ – poate semn că în regiunile
#cu multe start-up-uri, PIB-ul e momentan mai mic (dezvoltare în fază incipientă?)
