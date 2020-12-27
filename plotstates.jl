# plot states with reward location and 

theta=0:pi/50:2pi+pi/50;



using PyPlot
using PyCall

fig2 = figure("Line Collection Example")
#ax = PyPlot.axes(xlim = (-R,R),ylim=(-R,R))

# plot circle 
plot(R*cos.(theta),R*sin.(theta),ls="--",color=[169/255,169/255,169/255])
scatter(coordstates[1,:],coordstates[2,:])
plot(Xplatform[1].+r*cos.(theta),Yplatform[1].+r*sin.(theta),color=[169/255,169/255,169/255])

savefig("states.svg")

show()