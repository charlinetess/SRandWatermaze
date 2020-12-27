
using LinearAlgebra
using Statistics
using JLD2
using FileIO
using PyCall
# matplotlib.use("Agg")
# ENV["MPLBACKEND"]="tkagg"
using PyPlot
using Colors
using PyCall
@pyimport matplotlib.animation as anim
# anim=pyimport("matplotlib.animation") # I thnk new way of doing it 
using IJulia
zoom= pyimport("mpl_toolkits.axes_grid1.inset_locator") # for embeddded plots 
@pyimport matplotlib.projections as proj
@pyimport numpy as np



# 	
# 	            ,,                        ,...,,                                                ,...,,
# 	`7MM"""Mq.`7MM           mm         .d' ""db                                              .d' ""db                     mm
# 	  MM   `MM. MM           MM         dM`                                                   dM`                          MM
# 	  MM   ,M9  MM  ,pW"Wq.mmMMmm      mMMmm`7MM  .P"Ybmmm `7MM  `7MM  `7Mb,od8 .gP"Ya       mMMmm`7MM  `7Mb,od8 ,pP"Ybd mmMMmm
# 	  MMmmdM9   MM 6W'   `Wb MM         MM    MM :MI  I8     MM    MM    MM' "',M'   Yb       MM    MM    MM' "' 8I   `"   MM
# 	  MM        MM 8M     M8 MM         MM    MM  WmmmP"     MM    MM    MM    8M""""""       MM    MM    MM     `YMMMa.   MM
# 	  MM        MM YA.   ,A9 MM         MM    MM 8M          MM    MM    MM    YM.    ,       MM    MM    MM     L.   I8   MM
# 	.JMML.    .JMML.`Ybmd9'  `Mbmo    .JMML..JMML.YMMMMMb    `Mbod"YML..JMML.   `Mbmmd'     .JMML..JMML..JMML.   M9mmmP'   `Mbmo
# 	                                             6'     dP
# 	                                             Ybmmmd'
activities_all[150][findall(x->(x.==NaN),activities_all[150])]
activities_all[150][findall(x->!isnan(x),activities_all[150])]

global x
global y 
steps=1;
x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
y=zeros(1,length(x));
transpose!(y,x);   
global X,Y = np.meshgrid(x,y)

# draw a small circle around the iput location to check that activity is centered around input 
r=2.;
theta=0:0.001:2*pi;

ax = PyPlot.axes(xlim = (-R,R),ylim=(-R,R))

# homemadecolor=ColorMap("C", [RGB(153/255,204/255,255/255),RGB(75/255,102/255,125/255),RGB(0,0.5,0)]);


global argument=0:pi/50:2pi+pi/50;
global xmaze=R*cos.(argument);
global ymaze=R*sin.(argument);

i=150

plotact=activities_all[i];
maxi=maximum(plotact[findall(x->!isnan(x),plotact)])
mini=minimum(plotact[findall(x->!isnan(x),plotact)])
new_clim = (mini,maxi)
new_clim = (mini,10000)
cbar=colorbar()

# find location of the new upper limit on the color bar
loc_on_cbar = cbar.norm(new_clim[1])
cbar.ax.set_ylim(mini, 10000)

pcolormesh(x,x,plotact)#[1] 
plot(coordstates[:,indexinput1][1].+r.*cos.(theta),coordstates[:,indexinput1][2].+r.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])[1]
plot(coordstates[:,indexinput2][1].+r.*cos.(theta),coordstates[:,indexinput2][2].+r.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])[1] # goal input
plot(R*cos.(argument),R*sin.(argument),color="darkgrey")[1]  # maze 
	


# cbar.ax.set_ylim(0, loc_on_cbar)

show()



# 	
# 	                       ,,                                      ,,
# 	`7MMF'   `7MF'         db                                    `7MM           mm
# 	  MM       M                                                   MM           MM
# 	  MM       M ,pP"Ybd `7MM  `7MMpMMMb.  .P"Ybmmm     `7MMpdMAo. MM  ,pW"Wq.mmMMmm ,pP"Ybd
# 	  MM       M 8I   `"   MM    MM    MM :MI  I8         MM   `Wb MM 6W'   `Wb MM   8I   `"
# 	  MM       M `YMMMa.   MM    MM    MM  WmmmP"         MM    M8 MM 8M     M8 MM   `YMMMa.
# 	  YM.     ,M L.   I8   MM    MM    MM 8M              MM   ,AP MM YA.   ,A9 MM   L.   I8
# 	   `bmmmmd"' M9mmmP' .JMML..JMML  JMML.YMMMMMb        MMbmmd'.JMML.`Ybmd9'  `MbmoM9mmmP'
# 	                                      6'     dP       MM
# 	                                      Ybmmmd'       .JMML.


global x
global y 
steps=1;
x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
y=zeros(1,length(x));
transpose!(y,x);   
global X,Y = np.meshgrid(x,y)

# draw a small circle around the iput location to check that activity is centered around input 
r=2.;
theta=0:0.001:2*pi;

fig3 = plt.figure("MyFigure")#,figsize=(10,20))#onstrained_layout=true)

ax = PyPlot.axes(xlim = (-R,R),ylim=(-R,R))
ax.set_axis_off()
homemadecolor=ColorMap("C", [RGB(153/255,204/255,255/255),RGB(75/255,102/255,125/255),RGB(0,0.5,0)]);


global argument=0:pi/50:2pi+pi/50;
global xmaze=R*cos.(argument);
global ymaze=R*sin.(argument);





# global ax=gca()

# without labels 
global line1 = ax[:pcolormesh](X,Y,activities_all[1], shading="flat")#[1] 
global line2 = ax[:plot]([],[],linewidth=2,color=[169/255,169/255,169/255])[1] # start input 
global line3 = ax[:plot]([],[],linewidth=2,color=[250/255,128/255,114/255])[1] # goal input
global line4= ax[:plot]([],[],linewidth=0.5,color="darkgrey")[1]  # maze 
ax.set_axis_off()
cbar = fig.colorbar()
cbar[:set_visible_off]()


global maxi=maximum(activities_all[1][findall(x->!isnan(x),activities_all[1])])
global mini=minimum(activities_all[1][findall(x->!isnan(x),activities_all[1])])


# ax.spines["top"].set_color("none")
# ax.spines["right"].set_color("none")
# ax.spines["bottom"].set_visible("False")
# ax.spines["left"].set_visible("False")
# majors = [2*pi*l/parameters[:NA] for l in [0, 45, 90, 135,180]]; # x locations of the ticks 
# ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors)) 
# labels=["0°", "90°", "180°", "270°","360°"]
# ax.set_xticklabels(labels) # labels of the ticks 
# legend(loc="upper right")

# indexvector=indexvector+1;
# X,Y = np.meshgrid(x,y)
# global image 
# image=ax.pcolormesh(X,Y,eigenvector,vmin=-0.06,vmax=0.04)

# pcolormesh(x,x,act)
# colorbar()
# gca()[:grid](false);
# gca()[:set_axis_off]();
# #scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
# plot(coordstates[:,indexinput2][1].+radius.*cos.(theta),coordstates[:,indexinput2][2].+radius.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
# plot(coordstates[:,indexinput1][1].+radius.*cos.(theta),coordstates[:,indexinput1][2].+radius.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
# plot(,xmaze,ymaze);



# axbis.set_axis_off()





# Define the init function, which draws the first frame (empty, in this case)
function init()
    global line1
    global line2
    global line3
	global line4

	ax.set_axis_off()

	# ymax=maximum([maximum(data[indexrat][indextrial].historyUA[1]),maximum(parameters[:dt]/parameters[:τsa]*parameters[:εPCA]*data[indexrat][indextrial].historyinputPC[1]),maximum(parameters[:dt]/parameters[:τsa]*data[indexrat][indextrial].historyinputnoise[1])]);
	# ymax=ymax+ymax*2/100;
	# ymin=minimum([minimum(data[indexrat][indextrial].historyUA[1]),minimum(parameters[:dt]/parameters[:τsa]*parameters[:εPCA]*data[indexrat][indextrial].historyinputPC[1]),minimum(parameters[:dt]/parameters[:τsa]*data[indexrat][indextrial].historyinputnoise[1])]);
	# ymin=ymin-ymin*2/100;
	ax[:set_ylim]([-R,R])
	ax[:set_xlim ]([-R,R])
	ax[:set_axis_off]()
	# #scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
	# plot(coordstates[:,indexinput2][1].+radius.*cos.(theta),coordstates[:,indexinput2][2].+radius.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
	# plot(coordstates[:,indexinput1][1].+radius.*cos.(theta),coordstates[:,indexinput1][2].+radius.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
	# plot(,xmaze,ymaze);

		
	line1[:set_array](activities_all[1])# activity pcolormesh  
	line2[:set_data](coordstates[:,indexinput1][1].+r.*cos.(theta),coordstates[:,indexinput1][2].+r.*sin.(theta))# goal input 
    line3[:set_data](coordstates[:,indexinput2][1].+r.*cos.(theta),coordstates[:,indexinput2][2].+r.*sin.(theta))# start input 
	line4[:set_data](R*cos.(argument),R*sin.(argument))

    return (line1,line2,line3,line4,Union{})  # Union{} is the new word for None
end



function animate(i)
    global line1
    global line2
    global line3
	global line4


	 i=i+1


	ax.set_axis_off()

	# ymax=maximum([maximum(data[indexrat][indextrial].historyUA[1]),maximum(parameters[:dt]/parameters[:τsa]*parameters[:εPCA]*data[indexrat][indextrial].historyinputPC[1]),maximum(parameters[:dt]/parameters[:τsa]*data[indexrat][indextrial].historyinputnoise[1])]);
	# ymax=ymax+ymax*2/100;
	# ymin=minimum([minimum(data[indexrat][indextrial].historyUA[1]),minimum(parameters[:dt]/parameters[:τsa]*parameters[:εPCA]*data[indexrat][indextrial].historyinputPC[1]),minimum(parameters[:dt]/parameters[:τsa]*data[indexrat][indextrial].historyinputnoise[1])]);
	# ymin=ymin-ymin*2/100;
	ax[:set_ylim]([-R,R])
	ax[:set_xlim ]([-R,R])

	# #scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
	# plot(coordstates[:,indexinput2][1].+radius.*cos.(theta),coordstates[:,indexinput2][2].+radius.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
	# plot(coordstates[:,indexinput1][1].+radius.*cos.(theta),coordstates[:,indexinput1][2].+radius.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
	# plot(,xmaze,ymaze);
	plotact=activities_all[i][1:end-1, 1:end-1]# necessary to not split the plot - god knows why - found this on https://stackoverflow.com/questions/18797175/animation-with-pcolormesh-routine-in-matplotlib-how-do-i-initialize-the-data
	
	currentmaxi=maximum(activities_all[i][findall(x->!isnan(x),activities_all[i])])
	currentmini=minimum(activities_all[i][findall(x->!isnan(x),activities_all[i])])

	plotact=(maxi-mini)/(currentmaxi-currentmini)*plotact.+(maxi-(maxi-mini)/(currentmaxi-currentmini)*currentmaxi);

	# plotact=maxi*plotact/currentmaxi; # rescale 

	line1[:set_array](plotact)# activity pcolormesh  
	# new_clim = (mini,maxi)
	# # find location of the new upper limit on the color bar
	# loc_on_cbar = cbar.norm(new_clim[1])
	# cbar.ax.set_ylim(0, loc_on_cbar)
	line2[:set_data](coordstates[:,indexinput1][1].+r.*cos.(theta),coordstates[:,indexinput1][2].+r.*sin.(theta))# goal input 
    line3[:set_data](coordstates[:,indexinput2][1].+r.*cos.(theta),coordstates[:,indexinput2][2].+r.*sin.(theta))# start input 
	line4[:set_data](R*cos.(argument),R*sin.(argument))


    return (line1,line2,line3,line4, Union{})
end


mywriter = anim.FFMpegWriter()
# myanim = anim.FuncAnimation(fig3, animate, init_func=init, frames=length(activities_all), interval=200)
myanim = anim.FuncAnimation(fig3, animate, frames=length(activities_all), interval=200)

myanim[:save]("GloVidq=$(q)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon).mp4",writer=mywriter)





# 	
# 	                       ,,                                                            ,,
# 	`7MMF'   `7MF'         db                           `7MMF'                         `7MM
# 	  MM       M                                          MM                             MM
# 	  MM       M ,pP"Ybd `7MM  `7MMpMMMb.  .P"Ybmmm       MM  `7MMpMMMb.pMMMb.  ,pP"Ybd  MMpMMMb.  ,pW"Wq.`7M'    ,A    `MF'
# 	  MM       M 8I   `"   MM    MM    MM :MI  I8         MM    MM    MM    MM  8I   `"  MM    MM 6W'   `Wb VA   ,VAA   ,V
# 	  MM       M `YMMMa.   MM    MM    MM  WmmmP"         MM    MM    MM    MM  `YMMMa.  MM    MM 8M     M8  VA ,V  VA ,V
# 	  YM.     ,M L.   I8   MM    MM    MM 8M              MM    MM    MM    MM  L.   I8  MM    MM YA.   ,A9   VVV    VVV
# 	   `bmmmmd"' M9mmmP' .JMML..JMML  JMML.YMMMMMb      .JMML..JMML  JMML  JMML.M9mmmP'.JMML  JMML.`Ybmd9'     W      W
# 	                                      6'     dP
# 	                                      Ybmmmd'


fig = figure(figsize=(8,8))

function make_frame(i)
    imshow(activities_all[1+i],interpolation="none")
     gca()[:set_axis_off]()

end


mywriter = anim.FFMpegWriter()
# myanim = anim.FuncAnimation(fig3, animate, init_func=init, frames=length(data[indexrat][indextrial].historyUA), interval=200)


myanim = anim.FuncAnimation(fig, make_frame, frames=size(activities_all,1), interval=100)
myanim[:save]("test3.mp4",writer=mywriter)

# myanim[:save]("test3.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])}}

