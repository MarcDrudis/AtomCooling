# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:26:00 2021

@author: marcs
"""


import chart_studio.plotly as py
import plotly.graph_objs as go
#import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np



def doublePlot(x,y0,y1,title,xaxis,y0axis,y1axis,y0legend,y1legend):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=y0, name=y0legend),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1legend),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text=title
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text=xaxis)
    
    # Set y-axes titles
    fig.update_yaxes(title_text=y0axis, secondary_y=False)
    fig.update_yaxes(title_text=y1axis, secondary_y=True)
    
    return fig

def plotStd(x,y,std):
    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines'
        ),
        go.Scatter(
            x=np.concatenate((x , x[::-1])), # x, then x reversed
            y=np.concatenate((y+std,y[::-1]-std[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ])
    
    
    
    
    return fig

def plotquantile(x,y,qdown,qup,xaxis,yaxis,title,legend):
    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name='Average '+legend
        ),
        go.Scatter(
            x=np.concatenate((x , x[::-1])), # x, then x reversed
            y=np.concatenate((qup,qdown[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='10th to 90th percentile',
            showlegend=False
        )
    ])
    
    # Add figure title
    fig.update_layout(
        title_text=title 
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text=xaxis)
    
    # Set y-axes titles
    fig.update_yaxes(title_text=yaxis)
    
    
    return fig


def doubleplotquantile(x,y,y2,qdown,qup,xaxis,yaxis,yaxis2,title,legend,legend2):
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    # Add traces
    fig.add_trace(
       go.Scatter(x=x,y=y,line=dict(color='rgb(0,100,80)'),mode='lines',name='Average '+legend),
       secondary_y= False
        
       )
    
    fig.add_trace(
        go.Scatter(
            x=np.concatenate((x , x[::-1])), # x, then x reversed
            y=np.concatenate((qup,qdown[::-1])), # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='10th to 90th percentile',
            showlegend=False
        ),
        secondary_y= False
        )
    
    fig.add_trace(
        go.Scatter(x=x, y=y2, name=legend2,line=dict(color='rgb(255,0,0)')),
        secondary_y=True
    )
    
    # Add figure title
    fig.update_layout(
        title_text=title
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text=xaxis)
    
    # Set y-axes titles
    fig.update_yaxes(title_text=yaxis, secondary_y=False)
    fig.update_yaxes(title_text=yaxis2, secondary_y=True)
    
    return fig

#Specific plots
def plot_heuristic_dL(monty_plot,lq,hq):    
    
    mean_measure=[('module',None),('mean',None)]
    dL_mean=monty_plot.heuristic_dL(mean_measure,mean_measure)
    
    lq_measure=[('module',None),('quantile',lq)]
    dL_lq=monty_plot.heuristic_dL(lq_measure,lq_measure)
    
    hq_measure=[('module',None),('quantile',hq)]
    dL_hq=monty_plot.heuristic_dL(hq_measure,hq_measure)
    
    
    current_dL=monty_plot.deltaL
    t=np.arange(monty_plot.timesteps)*monty_plot.dt
    
    fig=plotquantile(x=1000*t,y=dL_mean,qdown=dL_lq,qup=dL_hq,xaxis='Time[ms]',yaxis='dL[Hz]',title='Heuristic dL',legend='Heuristic dL')
    fig.add_trace(go.Scatter(x=t*1000, y=current_dL, name='currentdL', line=dict(color='rgb(100,255,80)') ))
    return fig

def plot_temperature(monty,lq=0.1,hq=0.9):
    temperature=monty.observable('V',monty.temperature_measure)
    t=np.arange(monty.timesteps)*monty.dt
    
    Tm,Tl,Th=temperature.mean(axis=1),np.quantile(temperature,lq,axis=1),np.quantile(temperature,hq,axis=1)
    
    figV=plotquantile(t*1000,1000*Tm,1000*Tl,1000*Th,'Time[ms]','Temperature [mK]',"Poulation's Temperature",'temperature')
    figV.show()
    
def plots(monty,w=5,lq=0.1,hq=0.9,t=-1):   
    lq_measure=[('module',None),('quantile',lq)]
    hq_measure=[('module',None),('quantile',hq)]
    mean_measure=[('module',None),('mean',None)]
    last_average=[('module',None),('mean',None),('timestep',-1)]
    tt=np.arange(monty.timesteps)*monty.dt
    figX=doubleplotquantile(x=tt*1000,y=monty.observable('X',mean_measure,t)*1000,y2=monty.derivative_efficiency(w=w),qdown=monty.observable('X',lq_measure,t)*1000,qup=monty.observable('X',hq_measure,t)*1000,xaxis='Time[ms]',yaxis='Position[mm]',yaxis2='Particles leaving the box',title='Position ',legend='Position',legend2='Particles leaving(smoothed)')
    figX.show()


    figV=plotquantile(tt*1000,monty.observable('V',mean_measure,t),monty.observable('V',lq_measure,t),monty.observable('V',hq_measure,t),'Time[ms]','Speed[m/s]',"Poulation's speed",'speed')
    figV.add_trace(go.Scatter(x=tt*1000, y=monty.deltaL/monty.atom.k, name='Effective laser speed', line=dict(color='rgb(255,100,80)') ))
    figV.show()


    print('The final average Velocity is:{:.2f} m/s'.format(monty.observable('V',last_average,t)))
    print('The final average Distance from the Origin is:{:.3f} mm'.format(1000*monty.observable('X',last_average,t)))
    print("In the end we still have {:.1f}% of the particles".format(100*monty.efficiency()[-1]))
    
def TimeHist(monty_plot,observable,measure,xtitle,nbins=15,t=-1):   
    someT=monty_plot.observable(observable,measure,t)
    Selected_timesteps=np.linspace(0, monty_plot.timesteps-1, nbins).astype(int)
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in Selected_timesteps:
        fig.add_trace( px.histogram(x=someT[step],marginal='box').data[0] )
    
    # Make 0th trace visible
    fig.data[0].visible = True
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Time={Selected_timesteps[i]*monty_plot.dt*1000}ms"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders
    )
    fig.update_xaxes(title_text=xtitle,range=[0,someT.max()])
    fig.update_yaxes(title_text='Frequency')
    
    return fig


def TimeBox(monty_plot,observable,sampleN=1000,sampleT=20,t=-1):
    if sampleN==None:
        sampleN=monty_plot.N
    coorx=monty_plot.observable(observable,[('axis','x')],t)[:,:sampleN]
    coorx *= (observable=='X')*1000+(observable=='V')
    coory=monty_plot.observable(observable,[('axis','y')],t)[:,:sampleN]
    coory*= (observable=='X')*1000+(observable=='V')
    coorz=monty_plot.observable(observable,[('axis','z')],t)[:,:sampleN]
    coorz*= (observable=='X')*1000+(observable=='V')
    Selected_timesteps=np.linspace(0, monty_plot.timesteps-1,sampleT).astype(int)
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in Selected_timesteps:
        fig.add_trace(go.Box(x=coorz[step],name="Zaxis"))
        fig.add_trace(go.Box(x=coory[step],name="Yaxis"))
        fig.add_trace(go.Box(x=coorx[step],name="Xaxis"))
        
        
        
        
    # Make 0th trace visible
    fig.data[0].visible = True
    
    # Create and add slider
    steps = []
    for i in range(int(len(fig.data)/3)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Time={Selected_timesteps[i]*monty_plot.dt*1000}ms"}],  # layout attribute
        )
        step["args"][0]["visible"][3*i:3*i+3] = [True,True,True]  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders
    )
    
    if observable=='X':
        fig.update_xaxes(title_text="Position[mm]",range=[-1000*monty_plot.waist,1000*monty_plot.waist])
    
    else:
        bound=np.abs(np.concatenate((coorx,coory,coorz))).max()
        fig.update_xaxes(title_text="Velocity[m/s]",range=[-bound,bound])

     
    #fig.update_traces(boxpoints='all', jitter=0)
    fig.show()
    

def TimeCoords(monty_plot,observable='X',coordA='x',coordB='z',sampleN=1000,sampleT=5):
    coordinateA=monty_plot.observable(observable,[('axis',coordA)])[:,:sampleN]
    coordinateA*=(observable=='X')*1000+(observable=='V')
    
    coordinateB=monty_plot.observable(observable,[('axis',coordB)])[:,:sampleN]
    coordinateB*=(observable=='X')*1000+(observable=='V')
    
    
    Selected_timesteps=np.linspace(0, monty_plot.timesteps-1,sampleT).astype(int)
    
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in Selected_timesteps:
        fig.add_trace(go.Scatter(x=coordinateA[step],y=coordinateB[step],mode='markers',name='markers'))
        
        
    # Make 0th trace visible
    fig.data[0].visible = True
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Time={Selected_timesteps[i]*monty_plot.dt*1000}ms"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders
    )
    if observable=='X':
        fig.update_xaxes(title_text=f"{coordA} Position[mm]",range=[-1000*monty_plot.waist,1000*monty_plot.waist])
        fig.update_yaxes(title_text=f"{coordB} Position[mm]",range=[-1000*monty_plot.waist,1000*monty_plot.waist])
    else:
        bound=np.abs( np.concatenate((coordinateA,coordinateB)) ).max()
        fig.update_xaxes(title_text=f"{coordA} Velocity[m/s]",range=[-bound,bound])
        fig.update_yaxes(title_text=f"{coordB} Velocity[m/s]",range=[-bound,bound])

    fig.update_layout(autosize=False,width=500,height=500)
    fig.show()


def PhaseSpace(monty_plot,sampleN=1000,sampleT=5,t=-1):
    
    Position=monty_plot.observable('X',[('module',None),('multiply',1000)],t)[:,:sampleN]
    Velocity=monty_plot.observable('V',[('module',None)],t)[:,:sampleN]
    Position_average=monty_plot.observable('X',[('module',None),('multiply',1000),('mean',None)],t)
    Velocity_average=monty_plot.observable('V',[('module',None),('mean',None)],t)
    
    
    
    Selected_timesteps=np.linspace(0, monty_plot.timesteps-1,sampleT).astype(int)
    
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in Selected_timesteps:
        fig.add_trace(go.Scatter(x=Position[step],y=Velocity[step],mode='markers',name=f"{sampleN} samples"))
        fig.add_trace(go.Scatter(x=[Position_average[step]],y=[Velocity_average[step]],mode='markers',name='average',marker=dict(size=40)))
        
        
    # Make 0th trace visible
    fig.data[0].visible = True
    
    # Create and add slider
    steps = []
    for i in range(int(len(fig.data)/2)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Time={Selected_timesteps[i]*monty_plot.dt*1000}ms"}],  # layout attribute
        )
        step["args"][0]["visible"][2*i:2*i+2] = [True,True]  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Timestep: "},
       pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders
    )

    fig.update_xaxes(title_text=" Position[mm]",range=[0,np.sqrt(3)*1000*monty_plot.waist])
    bound=np.abs( Velocity ).max()
    fig.update_yaxes(title_text="Velocity[m/s]",range=[0,bound])

    fig.update_layout(autosize=False,width=300*np.sqrt(3)*1000*monty_plot.waist ,height=300*np.abs( Velocity[0] ).max())
    fig.show()
