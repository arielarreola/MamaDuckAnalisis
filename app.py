import streamlit as st #funciones generales de streamlit
from bokeh.plotting import figure as grafica #para mostrar graficas de lineas
import plotnine as p9 #pip install plotnine
from bokeh.models import ColumnDataSource
from PIL import Image #para abrir imagenes
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import pydeck as pdk
import datetime#libreria para usar formatos de fechas 
import json#libreria para usar json
import matplotlib.pyplot as plt
from pyvis import network as net
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from queue import PriorityQueue #libreria colas de prioridad
import math #Para los infinitos

image = Image.open('duck.png')
st.title("Reportes para Mama Duck")
col1, col2, col3 = st.beta_columns([1,45,1])#son columnas que maneja streamlit en la pagina
with col2:
    st.image(image, caption='Mama Duck',width=80)

#importamos el json
with open('ejemplo.json') as file:
    datajson = json.load(file)
jnodes=datajson.get("nodes")
jn1=jnodes[0]#esta variable guarda todo el json





#*
#*
## ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
## ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA

def DataJSONtoGraph(numnodo):#funcion para extraer datos de acuerdo con el formato del algoritmo de recorrido
  ismamaduck=False
  numcalls=0
  activetime=None
  numcalls=len(jn1.get(numnodo)[0]['history'])#ya esta listo numcalls
  numConected=len(jn1.get(numnodo)[0]['conections'])
  if numnodo=='1':
    ismamaduck=True
  adys=jn1.get(numnodo)[0]['conections']
  return int(numnodo),numConected,None,ismamaduck,numcalls,activetime,adys



def recorridonodos(n1):  # recibe una cadena de valores
    n0 = n1  # lo cofiguro como un nodo tipo puzzle
    Q = PriorityQueue()  # Q es una cola de prioridad
    aux = 0  # otro indice secundario de prioridad para Q
    Q.put((n0.f, aux, n0))  # la cola de prioridad se manejará mediante los f(n)
    visitados = []
    visitadoschart = []
    while not Q.empty():  # mientras la cola no este vacía
        u = Q.get()  # con el metodo get se guarda en u pero se quita de la cola el elemento
        u = u[2]  # porque una cola de prioridad almacena tuplas de prioridad,contador,nodo
        if u.tag not in visitados:
            visitados.append(u.tag)  # para evitar volverlo a visitar
            visitadoschart.append(u)
        ady = u.expand(u.ady1)  # expand me genera una lista de adyacencia, con heuristica y señalando a su padre, establece costo de 1 al generar neuvo nodo
        for v in ady:  # explorar los vecinos
 #           print("Heuristica de ", v.tag, " :", v.h)
            if v.tag not in visitados:  # si todavia no esta en visitados
                fp = v.h + v.g  # cálculo de funciones
                if fp < v.f:
                    v.f = fp
                    aux = aux + 1  # debo tener un entero antes de insertar un nodo en prioridad
                    Q.put((-(v.f), aux,
                           v))  # lo colocamos en la cola, para que en cada ciclo se evite agregar uno repetido
    return visitados, visitadoschart
def functionwhy(result):
  chart=list()
  ismd=""
  for u in result:
    if u.mamaduck==True:
      ismd="Sí"
    else:
      ismd="No"
    chart.append([u.tag,u.conexiones,ismd,u.h,u.ady1])
  df=pd.DataFrame(chart)
  df.columns = ["No. de nodo", "No. nodos conectados", "Es Mamaduck?","Llamadas recibidas","Adyacentes"]
  st.write(df)

  return df
class Nodo:
    w=None
    h = None  # heuristica
    f = math.inf  # f es infinito

    def __init__(self, tag, conexiones, padre, mamaduck, personasconect, tiempoactivos,ady1):  # inicializador
        self.tag = tag
        self.conexiones = conexiones
        self.padre = padre
        self.mamaduck = mamaduck
        self.tiempoactivos = tiempoactivos
        self.ady1 = ady1
        if padre is not None:  # si es un hijo
            self.g = padre.g + 1
        else:  # si es padre
            self.g = 0
            self.f = 0  # su f debe valer 0
        self.h = personasconect
    def expand(self, ady1):  # nos va a decir como explorar el grafo, obtiene la raiz en primera instancia
        ady = []  # aqui guardaremos los adyacentes que son objetos del tipo nodo
        for i in ady1:
          tag,conections,parent,mamaduck,people,timeactive,list1=DataJSONtoGraph(i)
          ady.append(Nodo(tag, conections, self, mamaduck, people, timeactive,list1))
        #  for i in ady1:
        #      ady.append(Nodo(ady1[i].conexiones,self,ady1[i].mamaduck,ady1[i].personasconect,ady1[i].tiempoativos))#creo nodo y lo guardo a la cola de adyacentes
        return ady  # retornamos los adyacentes o hijos del nodo expandido




def prepdatoLRML(e,f,g):
  max_item_index = f.index(max(f, key=int))
  namemergency=e[max_item_index]
  contador=0
  tuplapred=[]
  for i in range(len(e)):
    contador+=f[i]
  porcentaje=f[max_item_index]/contador
  if namemergency=='fire':
    tuplapred.append([porcentaje+1,g[0].get('firerisk')])
  elif namemergency=='medic':
    tuplapred.append([porcentaje,g[0].get('medicrisk')])
  elif namemergency=='medic':
    tuplapred.append([porcentaje+2,g[0].get('securityrisk')])
  elif namemergency=='sos':
    tuplapred.append([porcentaje+3,(g[0].get('other')+g[0].get('securityrisk')+g[0].get('medicrisk')+g[0].get('firerisk'))/4])
  return tuplapred

def prepdatoLRMLGen(e,f,g):
  max_item_index = f.index(max(f, key=int))
  namemergency=e[max_item_index]
  contador=0
  tuplapred=[]
  promfire=prommedic=promsec=promother=0
  promedios=list()#aqui se guardará el promedio de riesgos de todos los nodos
  for elem in range(len(g)):
    promfire+=g[elem].get('firerisk')
    prommedic+=g[elem].get('medicrisk')
    promsec+=g[elem].get('securityrisk')
    promother+=g[elem].get('other')
  promedios.append(promfire/len(g))
  promedios.append(prommedic/len(g))
  promedios.append(promsec/len(g))
  promedios.append(promother/len(g))
  for i in range(len(e)):
    contador+=f[i]
  porcentaje=f[max_item_index]/contador
  if namemergency=='fire':
    tuplapred.append([porcentaje+1,promedios[0]])
  elif namemergency=='medic':
    tuplapred.append([porcentaje,promedios[1]])
  elif namemergency=='medic':
    tuplapred.append([porcentaje+2,promedios[2]])
  elif namemergency=='sos':
    tuplapred.append([porcentaje+3,(promedios[3]+promedios[2]+promedios[1]+promedios[0])/4])
  return tuplapred


strindices=[]
for i in range(len(jn1)):
  strindices.append(str(i+1))#esta ya puede utilizarse para otras funciones


def obtencionCoords():
  #lat y lon 1 es para coordenadas de llamadas
  #lat  lon2 son para coordenadas de NODOS
  list1=list()
  list2=list()
  for s in strindices:
    for i in range(len(jn1.get(str(s))[0]['history'])):
      auxlat=jn1.get(str(s))[0]['history'][i]['localization'][0]['lat']
      auxlon=jn1.get(str(s))[0]['history'][i]['localization'][0]['long']
      list1.append([auxlat,auxlon])

    list2.append([jn1.get(str(s))[0]['localization'][0]['lat'],jn1.get(str(s))[0]['localization'][0]['long']])
  a1=np.array(list1)
  a2=np.array(list2)
  #print(a1.reshape(-2,2))
  df1 = pd.DataFrame(
  a1,
  columns=['lat', 'lon'])
  #return df1, df2
  #print(a1.reshape(-2,2))
  df2 = pd.DataFrame(
  a2,
  columns=['lat', 'lon'])
  #return df1, df2
  return df1,df2


def obtencionlistasJS(numnodo):
  jnemergency=list()
  jnumsoc=list()#cuenta las ocurrencias de horas
  jhours=list()
  jndate=list()
  jrisks=list()
  varisk=jn1.get(numnodo)[0]['risks'][0]
  jrisks.append(varisk)
  #para jhours
  #para jnemergency
  for i in range(len(jn1.get(numnodo)[0]['history'])):
    varauxem=jn1.get(numnodo)[0]['history'][i]['emergency']
    jnemergency.append(varauxem)
    varauxda=jn1.get(numnodo)[0]['history'][i]['date']
    jndate.append(varauxda)
    varauxhr=jn1.get(numnodo)[0]['history'][i]['hour']
    varauxhr=int(varauxhr[0:2])
    jhours.append(varauxhr)
  #preproc no repetidos en jhours
  jhoursp=[]
  for item in jhours:
      if item not in jhoursp:
          jhoursp.append(item)
  #para jnumsoc
  for item in jhoursp:
    jnumsoc.append(jhours.count(item))
  #para jndate yconvertir a datetime
  dtdate=list()
  for y in range(len(jndate)):
    dtdate.append(datetime.datetime.strptime(str(jndate[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  #para listar tipos de emergencia no repetidos y contarlos
  nremerg=[]#emergencias de cada tipo no repetidas
  jnemergency2=list()#una copia para tener una lista de strs
  countemetype=list()#aqui se depositan los numeros de ocurrencia de cada emergencia segun el orden de nremerg
  for a in jnemergency:
    if len(a)>1:
      jnemergency2.append(str(a[0]))
      jnemergency2.append(str(a[1]))
    else:
      jnemergency2.append(str(a[0]))

  for item in jnemergency2:
      if item not in nremerg:
          nremerg.append(item)
  #print(nremerg)
  #print(jnemergency)
  ##print(nremerg)
  #para obtener num de datos
  for item in nremerg:
    countemetype.append(jnemergency2.count(item))
  ##print(countemetype)

  return dtdate,jnemergency2,jhoursp,jnumsoc,nremerg,countemetype,jrisks
#en este orden devuelve...
#fechas en formato datetime (solo fechas, no horas)
#emergencias por tipo (incluyendo si aplica mas de un tipo de emergencia)formato arreglo de [[],[],...]
#horas en las que se declararon las emergencias (sin repetirse)
#numero de ocurrencias de las horas de jhoursp
#tipo de emergencias en formato string y sin repetirse
#numero de emergencias por cada tipo (correspondiendo al orden presentado en nremerg)
def obtencionlistasJS3():
  jnemergency=list()
  jnumsoc=list()#cuenta las ocurrencias de horas
  jhours=list()
  jndate=list()
  jrisks=list()
  #para jhours
  #para jnemergency
  for numnodo in strindices:
      varisk=jn1.get(numnodo)[0]['risks'][0]
      jrisks.append(varisk)


  for numnodo in strindices:
    for i in range(len(jn1.get(str(numnodo))[0]['history'])):
      varauxem=jn1.get(str(numnodo))[0]['history'][i]['emergency']
      jnemergency.append(varauxem)
      varauxda=jn1.get(str(numnodo))[0]['history'][i]['date']
      jndate.append(varauxda)
      varauxhr=jn1.get(str(numnodo))[0]['history'][i]['hour']
      varauxhr=int(varauxhr[0:2])
      jhours.append(varauxhr)
  #preproc no repetidos en jhours
  jhoursp=[]
  for item in jhours:
      if item not in jhoursp:
          jhoursp.append(item)
  #para jnumsoc
  for item in jhoursp:
    jnumsoc.append(jhours.count(item))
  #para jndate yconvertir a datetime
  dtdate=list()
  for y in range(len(jndate)):
    dtdate.append(datetime.datetime.strptime(str(jndate[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  #para listar tipos de emergencia no repetidos y contarlos
  nremerg=[]#emergencias de cada tipo no repetidas
  jnemergency2=list()#una copia para tener una lista de strs
  countemetype=list()#aqui se depositan los numeros de ocurrencia de cada emergencia segun el orden de nremerg
  for a in jnemergency:
    if len(a)>1:
      jnemergency2.append(str(a[0]))
      jnemergency2.append(str(a[1]))
    else:
      jnemergency2.append(str(a[0]))

  for item in jnemergency2:
      if item not in nremerg:
          nremerg.append(item)
  #print(nremerg)
  #print(jnemergency)
  ##print(nremerg)
  #para obtener num de datos
  for item in nremerg:
    countemetype.append(jnemergency2.count(item))
  ##print(countemetype)

  return dtdate,jnemergency2,jhoursp,jnumsoc,nremerg,countemetype,jrisks

def obtencionlistasJS2():
  typemergency=list()
  jdays=list()#cuenta las ocurrencias de horas
  jhours=list()
  jnums=list()
  #para jhours
  #para jnemergency
  for s in strindices:
    for i in range(len(jn1.get(str(s))[0]['history'])):
      ad=jn1.get(str(s))[0]['history'][i]['date']
      jdays.append(ad)
  jdaystab=list()
  for e in jdays:
    if e not in jdaystab:
      jdaystab.append(e)
  for item in jdaystab:
    jnums.append(jdays.count(item))
  dtdate=list()
  for y in range(len(jdaystab)):
    dtdate.append(datetime.datetime.strptime(str(jdaystab[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  return dtdate,jnums
#devuelve en ese orden fechas convertidas, numeros de ocurrencia
def regresionLinealNumEmergen(X, Y, numuser,numnodo):
    X = X.reshape(-1, 1)
    # cada elemento solo tiene un feature
    Y = Y.reshape(-1, 1)
    # Modelo de regresion lineal
    # transpuesta de x
    xtran = np.transpose(X)  # primero ahcemos la transpuesta de X

    # Producto punto
    xtran_x = np.linalg.inv(xtran.dot(X))  # inversa del producto punto en la transpuesta de X
    xtran_y = xtran.dot(Y)  # producto punto de la transpuesta de X con la Y
    W = xtran_x.dot(xtran_y)  # producto punto de la transpuesta de X con la transpuesta de Y
    y_pred = X * W  # una variable a predecir utilizando el peso
    plt.scatter(X, Y)
    plt.plot(X, y_pred, color='g')

    test = numuser * W
    plt.scatter(numuser, test, color="y")
    plt.scatter(X, Y)
    plt.plot(X, y_pred, color='g')
    prediccion = W[0] * numuser
    st.write("Segun la hora recibida: ", numuser, ", se predijo este número de emergencias:", round(prediccion[0]),
          "Esos son los numeros de casos probables en el nodo",numnodo)
    st.write("*Prediccion realizada mediante el modelo de regresión lineal*")



def plothour_vs_numcalls(x,y,color1):
  infonumcalls=[]
  for i in range(len(c)):
    infonumcalls.append([x[i],y[i]])
  frameinfo=pd.DataFrame(infonumcalls,columns=['Hora','Num. emergencias'])
  if st.checkbox('Mostrar tabla de número de emergencias'):
    st.table(frameinfo)
  st.header('Emergencias acumuladas mostradas por hora de ocurrencia')
  dotgraph = p9.ggplot(data=frameinfo,
                        mapping=p9.aes(x='Hora', y='Num. emergencias'))

  st.pyplot(p9.ggplot.draw(dotgraph + p9.geom_line(color=color1)))
    #muestra una lámina graficando este tipo de datos

def plottypecalls(x,y,color1):
    p = grafica(
    title='Tipo de emergencias',
    x_axis_label='Emergencia',
    y_axis_label='Cantidad')

    p.line(x, y, legend_label='Numero de emergencias', line_width=2,color=color1)
    st.bokeh_chart(p, use_container_width=True)
    #muestra una lámina graficando este tipo de datos


def RegresionLogML(predict):

  '''porcentaje + 0= medic
  porcentaje + 1=fire
  porcentaje +2=security
  porcentaje +3=sos
  porcentaje +4=refuge
  porcentaje +5= others'''
  X=[[0.8,6],[0.8,7],[0.8,8],[0.8,9],[0.8,10],
    [0.7,6],[0.7,7],[0.7,8],[0.7,9],[0.7,10],
    [0.6,6],[0.6,7],[0.6,8],[0.6,9],[0.6,10],
    [0.9,6],[0.9,7],[0.9,8],[0.9,9],[0.9,10],
    [1,6],[1,7],[1,8],[1,9],[1,10],
    [1.6,6],[1.6,7],[1.6,8],[1.6,9],[1.6,10],
    [1.7,6],[1.7,7],[1.7,8],[1.7,9],[1.7,10],
    [1.8,6],[1.8,7],[1.8,8],[1.8,9],[1.8,10],
    [1.9,6],[1.9,7],[1.9,8],[1.9,9],[1.9,10],
    [2,6],[2,7],[2,8],[2,9],[2,10],
    [2.6,6],[2.6,7],[2.6,8],[2.6,9],[2.6,10],
    [2.7,6],[2.7,7],[2.7,8],[2.7,9],[2.7,10],
    [2.8,6],[2.8,7],[2.8,8],[2.8,9],[2.8,10],
    [2.9,6],[2.9,7],[2.9,8],[2.9,9],[2.9,10],
    [3,6],[3,7],[3,8],[3,9],[3,10],
    [3.6,6],[3.6,7],[3.6,8],[3.6,9],[3.6,10],
    [3.7,6],[3.7,7],[3.7,8],[3.7,9],[3.7,10],
    [3.8,6],[3.8,7],[3.8,8],[3.8,9],[3.8,10],
    [3.9,6],[3.9,7],[3.9,8],[3.9,9],[3.9,10],
    [4,6],[4,7],[4,8],[4,9],[4,10],
    ]
  y=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
  df=pd.DataFrame(X,columns=["%+type","Risk index"])
  y=np.array(y)
  df['Propenso a']=y
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)#aqui se pone en practica que el testing solo sera 20%
  model=LogisticRegression().fit(X_test,y_test)
  y_pred=model.predict(X_test)
  score=accuracy_score(y_test,y_pred)#aqui brinda el procentaje de aprendizaje adquirido con el train
  y_pred=model.predict(predict)
  if y_pred[0]==0:
    nem='medic'
  elif y_pred[0]==1:
    nem='fire'
  elif y_pred[0]==2:
    nem='security'
  elif y_pred[0]==3:
    nem='sos'
  else:
    nem='other'#falta entrenarlo con otros tipos de emergencia aparte de losmencionados
  st.write("Para el presente nodo, se predice que es más vulnerable a emergencias de tipo: ",nem)
  st.write("*Se utilizó el método de Regresión logística con un aprendizaje de ",score*100,"%*")
  st.success("Se ha completado el cálculo con éxito")






## FIN ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
## FIN ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
#*
#*







nodoseleccionado = st.radio("Seleccione un nodo",
('Mama Duck', 'Nodo 2', 'Nodo 3', 'Nodo 4','General'))
if nodoseleccionado=='Mama Duck':
  st.header('Análisis de los datos del Nodo Mama Duck')
  a,b,c,d,e,f,g=obtencionlistasJS('1')
#quiero intentar analizar equis tipo de llamada mas probable
  
  plothour_vs_numcalls(c,d,'blue')

#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
  user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
  for x in user_input:
    regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
    st.success("Se ha completado el análisis para la hora seleccionada")
  st.header('Análisis de emergencias recurrentes')
  if st.button("Obtener análisis de emergencia más probable"):
    RegresionLogML(prepdatoLRML(e,f,g))

    

elif nodoseleccionado=='Nodo 2':
  st.header('Análisis de los datos del Nodo 2')
  a,b,c,d,e,f,g=obtencionlistasJS('2')
#quiero intentar analizar equis tipo de llamada mas probable
  plothour_vs_numcalls(c,d,'red')
#predTipoEmergencia(x,y)
  #if st.button("Estimar no. de emergencias dada una hora", key=None, 
  #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
  user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
  for x in user_input:
      regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
      st.success("Se ha completado el análisis para la hora seleccionada")
  st.header('Análisis de emergencias recurrentes')
  if st.button("Obtener análisis de emergencia más probable"):
    RegresionLogML(prepdatoLRML(e,f,g))

elif nodoseleccionado=='Nodo 3':
    st.header('Análisis de los datos del Nodo 3')
    a,b,c,d,e,f,g=obtencionlistasJS('3')
#quiero intentar analizar equis tipo de llamada mas probable
    plothour_vs_numcalls(c,d,'brown')
#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
        regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
        st.success("Se ha completado el análisis para la hora seleccionada")
    st.header('Análisis de emergencias recurrentes')
    if st.button("Obtener análisis de emergencia más probable"):
      RegresionLogML(prepdatoLRML(e,f,g))
elif nodoseleccionado=='Nodo 4':
    st.header('Análisis de los datos del Nodo 4')
    a,b,c,d,e,f,g=obtencionlistasJS('4')
#quiero intentar analizar equis tipo de llamada mas probable
    plothour_vs_numcalls(c,d,'purple')
#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
        regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
        st.success("Se ha completado el análisis para la hora seleccionada")
    st.header('Análisis de emergencias recurrentes')
    if st.button("Obtener análisis de emergencia más probable"):
      RegresionLogML(prepdatoLRML(e,f,g))
#voy a desplegar el historial dellamadas en una tabla
#voy a mostrar el numero de llamadas por dia

elif nodoseleccionado=='General':
  st.header('Análisis general de la red Mama Duck')
  xi,yi,c,d,e,f,g=obtencionlistasJS3()
  #x=fecha, y=emergencias sin repetir
  plothour_vs_numcalls(c,d,'red')
  user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
  for x in user_input:
      regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
      st.success("Se ha completado el análisis para la hora seleccionada")
  listdt=list()
  a,b=obtencionlistasJS2()

  for x in range(len(a)):
    stra=str(a[x])
    listdt.append([stra[5:10],b[x]])
  df=pd.DataFrame(listdt,columns=['Fecha','Emergencias'])
  st.header("Historial de emergencias por fecha")
  dotgraph = p9.ggplot(data=df,
                          mapping=p9.aes(x='Fecha', y='Emergencias'))

  st.pyplot(p9.ggplot.draw(dotgraph +p9.geom_point(color='green',alpha=0.5,size=2.7)))
  #voy a mostrar el numero de llamadas por hora dado un dia
  #mostrar estadistica de tipo de accidente--graficas
  #mostrar warnings del nodo--lo que hizo alex de acuerdo con el json recibido
  #mostrar emergencias comunes--haciendo abstraccion de los datos recibidos
  #predecir posibles accidentes totales--usando el algoritmo de IRIS (visto en clases de ML)
  st.header('Análisis de emergencias recurrentes globales')
  if st.button("Análisis de emergencia más probable"):
    RegresionLogML(prepdatoLRMLGen(e,f,g))

  st.header("Mapa de nodos y emergencias")
  st.write("Hexagonos: ubicacion de nodos, Círculo verde: casos de emergencia")
  df1,df2=obtencionCoords()
  if st.checkbox('Mostrar tabla de coordenadas de emergencia'):
    st.write(df1)
  if st.checkbox('Mostrar tabla de coordenadas de Nodos'):
    st.write(df2)

  st.pydeck_chart(pdk.Deck(
  map_style='mapbox://styles/uberdata/cjoqbbf6l9k302sl96tyvka09',
  initial_view_state=pdk.ViewState(
      latitude=20.63494981128319,
      longitude=-103.40648023281342,
      zoom=17,
      pitch=50,
  ),
  
  layers=[
      pdk.Layer(
          'HexagonLayer',
          data=df2,
          get_position='[lon, lat]',
          #colorRange='[0, 240,255,255]',
          radius=3,
          elevation_scale=4,
          elevation_range=[0, 10],
          pickable=True,
          extruded=True,
      ),
      pdk.Layer(
          'ScatterplotLayer',
          data=df1,
          get_position='[lon, lat]',
          get_color='[100, 230, 0, 160]',
          get_radius=2,
      ),
  ],
  ))
  st.title("Nodos receptores Mama Duck")
  st.write("A continución se muestra la distribución de nodos de Mama Duck, sus interconexiones e información")
  g=net.Network(height='400px', width='60%')
  colacolores=["green","red","yellow","blue"]
  for i in range(len(jn1)):
    #issue: no detecta los saltos de linea
    g.add_node(i+1,title=jn1.get(str(strindices[i]))[0].get('name')+
    """\n"""+ """Status: """+jn1.get(str(strindices[i]))[0].get('status')+ """
    Risk index:"""+str(jn1.get(str(strindices[i]))[0].get('risks')[0]),color=colacolores.pop(),borderWidthSelected=3,labelHighlightBold=True)
      
  for sti in strindices:
    for s in range(len(jn1.get(sti)[0]['conections'])):
      aux=jn1.get(sti)[0]['conections']             
      g.add_edge(int(sti),int(aux[s]),color='black')
  

  g.save_graph('graph.html')
  HtmlFile=open('graph.html','r',encoding='utf-8')
  sourceCode=HtmlFile.read()
  components.html(sourceCode,height=400,width=1500)
  st.header("Monitoreo de nodos")
  tag,conections,parent,mamaduck,people,timeactive,list1=DataJSONtoGraph('1')

  n1=Nodo(tag,conections,parent,mamaduck,people,timeactive,list1)
  st.write("Se esta calculando el algoritmo...")
  res,restab=recorridonodos(n1)#iniciamos el algoritmo de A estrella en una variable de objeto
  st.write("El Nodo: %d "%res[1],"requiere ser monitoreado prioritariamente. Seleccione 'Más detalles' para más información")#en res 1 está el nodo elegido como prioritario, esta función imprimirá el porqué primero este y porque los demas
  st.header("Orden de prioridad")
  st.header(res)
  if st.button("Más detalles..."):
      if not restab and not res:
        st.write("Primero presione en el botón de Monitoreo de nodos para presentarle detalles")
      else:
        functionwhy(restab)

#SIN FUNCIONALIDAD POR AHORA PERO PUEDEN SERVIR MAS ADELANTE
#st.header('Histórico por día')
#st.date_input("Seleccione la fecha para consultar: ", 
#value=None, min_value=datetime.date(2021,4,10), max_value=None, key=None, help="Presione sobre la barra para desplegar el calendario")
    #mostrar emergencias por hora
#st.button("Estimar no. de emergencias dada una hora", key=None,  help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas")

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)
#
# st.pyplot(fig)

#st.time_input("Escriba la hora a consultar", value=None, key=None, help=None)