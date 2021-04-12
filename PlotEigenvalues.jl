using LightGraphs
using LinearAlgebra
using PyPlot
using Colors
using PyCall
using LaTeXStrings
@pyimport numpy as np
@pyimport matplotlib.patches as patch

# load functions: 
# to define states : 
# The algorithm places n points, of which the kth point is put at distance sqrt(k-1/2) from the boundary (index begins with k=1), and with polar angle 2*pi*k/phi^2 where phi is the golden ratio. Exception: the last alpha*sqrt(n) points are placed on the outer boundary of the circle, and the polar radius of other points is scaled to account for that. This computation of the polar radius is done in the function radius.
function  radius(k,n,b) # k index of point on the boundary, n number of total points, b number of boundary points
    if k>n-b
        r = 1;            # put on the boundary
    else
        r = sqrt(k-1/2)/sqrt(n-(b+1)/2);     # computation of radius of the different points 
    end
end


# sunflower seed arrangement :
function sunflower(n, R, alpha)   # n number of centers,
    # alpha is indicating how much one cares about the evenness of boundary , chose 2 to have nice trade off
    # R is the radius of the circle in cm
    r=zeros(1,n);
    theta=zeros(1,n);
    b = round(alpha*sqrt(n));      # number of boundary points
    phi = (sqrt(5)+1)/2;           # golden ratio
    for k=1:n
        r[k] = R*radius(k,n,b); # computation of the radius of each point 
        theta[k] = 2*pi*k/phi^2; # computation of the angle of each point 
        #plot(r*cos.(theta), r*sin.(theta), "m");
    end
    X=r.*cos.(theta); 
    Y=r.*sin.(theta);
    return vcat(X, Y)
end

# activation function
function activation(g,x)
    return g*max(0.0,x)
end


function complexactivation(x)
    return log(1+exp(x))
end

# potential param to play with q=$(q)_$(k)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon) 

# first trial on 3 param : sigma, q, and freeparameter 
sigma=2.5;
sigma2=4;

R= 100; # Radius of the circle in cm
r=5;# Radius of the platform  in cm
# parameters :
gamma=1; # discount factor
numberofneurons=500; # number of neurons 
numberofinputs=30; # number of inputs
# numberofneighbours=8; # Is the probability distribution concerns only the nearest neighbours and is 0 elsewhere 
## Create points equally distant throughout the environment 
numberofstates=1000; # number of points 
# define state coordinates 
coordstates=sunflower(numberofstates,R,2); # this gives the coordinates as a numberofneurons*2 array ordered depending on their radius 
# order states as a function of their X and Y value 
# coordstates=sortslices(coordstates,dims=2,by=x->(x[2],x[1])); # this gives the coordinates as a numberofneurons*2 array , the first states having the lowest y value
## INIT ACTIVITY :
# timestep
dt=0.001; # timestep in seconds
tau=1; # time constant of neuron in second
# define gain factor for the activity :
g=0.5; # testing the sensitivity to the slope of the activation function


epsilon=0.05; #  the timescale on which the network activity fades.
alpha=0.05; # input strength 
delayinput=2; # how many tau do we initiate the activity - 5 is bit too long 
delayend=8; # how long do we run the thing - 30 is too long 


### parameter for the input / goal state : we can either use the platform and start positions used during the experiment:
# Potential positions of the platform : 
Xplatform=[0.3,0,-0.3,0,0.5,-0.5,0.5,-0.5].*R; # in cm
Yplatform=[0,0.3,0,-0.3,0.5,0.5,-0.5,-0.5].*R;# in cm

# Potential Starting positions of the rat :
Xstart=[0.95,0,-0.95,0].*R; # East, North, West, South
Ystart=[0,0.95,0,-0.95].*R;

#indexstart=rand(1:4);
#indexplatform=rand(1:length(Xplatform));
indexstart=4;
indexplatform=5;


# find index of the first input state = the nearest state of the official starting points 
indexinput1=findall((minimum(sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1))[:]);
# find index of the second input state = the nearest state of the official platform point  
indexinput2=findall((minimum(sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1))[:]);

R= 100; # Radius of the circle in cm
r=5;# Radius of the platform  in cm
# parameters :
gamma=1; # discount factor
numberofneurons=500; # number of neurons 
numberofinputs=30; # number of inputs
# numberofneighbours=8; # Is the probability distribution concerns only the nearest neighbours and is 0 elsewhere 
## Create points equally distant throughout the environment 
numberofstates=1000; # number of points 
# define state coordinates 
coordstates=sunflower(numberofstates,R,2); # this gives the coordinates as a numberofneurons*2 array ordered depending on their radius 
# order states as a function of their X and Y value 
# coordstates=sortslices(coordstates,dims=2,by=x->(x[2],x[1])); # this gives the coordinates as a numberofneurons*2 array , the first states having the lowest y value
## INIT ACTIVITY :
# timestep
dt=0.001; # timestep in seconds
tau=1; # time constant of neuron in second
# define gain factor for the activity :
g=0.5; # testing the sensitivity to the slope of the activation function


epsilon=0.05; #  the timescale on which the network activity fades.
alpha=0.05; # input strength 
delayinput=2; # how many tau do we initiate the activity - 5 is bit too long 
delayend=8; # how long do we run the thing - 30 is too long 


### parameter for the input / goal state : we can either use the platform and start positions used during the experiment:
# Potential positions of the platform : 
Xplatform=[0.3,0,-0.3,0,0.5,-0.5,0.5,-0.5].*R; # in cm
Yplatform=[0,0.3,0,-0.3,0.5,0.5,-0.5,-0.5].*R;# in cm

# Potential Starting positions of the rat :
Xstart=[0.95,0,-0.95,0].*R; # East, North, West, South
Ystart=[0,0.95,0,-0.95].*R;

#indexstart=rand(1:4);
#indexplatform=rand(1:length(Xplatform));
indexstart=4;
indexplatform=5;
# find index of the first input state = the nearest state of the official starting points 
indexinput1=findall((minimum(sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1))[:]);
# find index of the second input state = the nearest state of the official platform point  
indexinput2=findall((minimum(sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1))[:]);


#   
#   
#   
#   
#    .gP"Ya `7MMpMMMb.`7M'   `MF'
#   ,M'   Yb  MM    MM  VA   ,V
#   8M""""""  MM    MM   VA ,V
#   YM.    ,  MM    MM    VVV
#    `Mbmmd'.JMML  JMML.   W
#   
#   

# first trial we consider an exponential transformation of the distances :
P=zeros(numberofstates,numberofstates);
P2=zeros(numberofstates,numberofstates);
for i=1:numberofstates
   for j=1:numberofstates
        P[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
        P2[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
   end
        P[i,:]=exp.(-P[i,:].^2/(2*sigma^2))./sum(exp.(-P[i,:].^2/(2*sigma^2))); # normalize it to obtain a probability distribution
        P2[i,:]=exp.(-P2[i,:].^2/(2*sigma2^2))./sum(exp.(-P2[i,:].^2/(2*sigma2^2))); # normalize it to obtain a probability distribution
end
eigenvalues=eigvals(P);
eigenvaluesbis=sort(eigenvalues,  rev=true);
# order eigenvectors
ordering=[findall(x->eigenvaluesbis[k]==x,eigenvalues)[1] for k in 1:numberofstates]; # 
# check that we got this right :
if !(eigenvaluesbis==eigenvalues[ordering])
    println("something is wrong with the order")
    #break
end
eigenvalues2=eigvals(P2);
eigenvaluesbis2=sort(eigenvalues2,  rev=true);
# order eigenvectors
ordering2=[findall(x->eigenvaluesbis2[k]==x,eigenvalues2)[1] for k in 1:numberofstates]; # 
# check that we got this right :
if !(eigenvaluesbis2==eigenvalues2[ordering2])
    println("something is wrong with the order")
    #break
end
#
#                ,,
#              `7MM           mm
#                MM           MM
#     `7MMpdMAo. MM  ,pW"Wq.mmMMmm
#       MM   `Wb MM 6W'   `Wb MM
#       MM    M8 MM 8M     M8 MM
#       MM   ,AP MM YA.   ,A9 MM
#       MMbmmd'.JMML.`Ybmd9'  `Mbmo
#       MM
#     .JMML.


fig = figure("Test plot latencies",figsize=(9,9))
ax = fig[:add_subplot](1,1,1)


xlabel("Index eigenvalue",fontsize=22)
ylabel("Power",fontsize=22)         

NumberEigen=500; # cnumber of eigenvalues that we want to take into account  
TimePower=[1 2 8 16 50] # We will show the eigenvalues of P^t and their evolution 
colors=["firebrick","peru", "forestgreen","royalblue","darkviolet"];

pright=[];
pleft=[];
for i=1:length(TimePower)
p1,=plot(1:1:NumberEigen,eigenvaluesbis[1:NumberEigen].^TimePower[i],label=latexstring("t^{$(TimePower[i])}"),color=colors[i],"-",lw=3)
p2,=plot(1:1:NumberEigen,eigenvaluesbis2[1:NumberEigen].^TimePower[i],label=latexstring("t^{$(TimePower[i])}"),color=colors[i],":",lw=3,alpha=0.8)
push!(pright,p1)
push!(pleft,p2)
end
p5, = plot([0], marker="None",
           linestyle="None", label="dummy-tophead")
p7, = plot([0],  marker="None",
           linestyle="None", label="dummy-empty")

# categories = [latexstring("\$\\sigma=$(sigma1)\$"), latexstring("\$\\sigma=$(sigma2)\$")]

categories = [latexstring("\$P^{$(TimePower[i])}\$") for i=1:length(TimePower)]

# leg3 = legend(vcat(p5,p[1:length(TimePower)],p5,p[length(TimePower)+1,end]),vcat([latexstring("\$\\sigma=$(sigma)\$")] , categories , [latexstring("\$\\sigma=$(sigma2)\$")] , categories),loc=2, ncol=2) # Two columns, vertical group labels

# leg3 = legend(vcat(p5,pright,p5,pleft),vcat([latexstring("\$\\sigma=$(sigma)\$")] , categories , [latexstring("\$\\sigma=$(sigma2)\$")] , categories),loc=2, ncol=2) # Two columns, vertical group labels

leg4 = legend(vcat(p5,pright,p5,pleft),vcat([latexstring("\$\\sigma=$(sigma)\$")] , categories , [latexstring("\$\\sigma=$(sigma2)\$")] , categories),loc=1, ncol=2) # Two columns, vertical group labels


# gca().add_artist(leg3)

# legend()

mx = matplotlib.ticker.MultipleLocator(100) # Define interval of minor ticks
ax.xaxis.set_major_locator(mx) # Set interval of minor ticks

ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["bottom"].set_visible("False")
ax.spines["left"].set_visible("False")

xmin, xmax = ax.get_xlim() 
ymin, ymax = ax.get_ylim()
ax[:set_ylim]([0,ymax])
ax[:set_xlim]([0,xmax])
xmin, xmax = ax.get_xlim() 
ymin, ymax = ax.get_ylim()
# get width and height of axes object to compute 
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height
# manual arrowhead width and length
hw = 1/20*(ymax-ymin) 
hl = 1/20*(xmax-xmin)
lw = 1 # axis line width
ohg = 0.3 # arrow overhang
# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

ax.arrow(xmin, ymin, xmax-xmin, 0.,length_includes_head= "True", fc="k", ec="k", lw = lw,head_width=hw, head_length=hl, overhang = ohg,  clip_on = "False") 

ax.arrow(xmin, ymin, 0., ymax-ymin,length_includes_head= "True", fc="k", ec="k", lw = lw, head_width=yhw, head_length=yhl, overhang = ohg,  clip_on = "False")

SMALL_SIZE = 10
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# savefig("megatest.png")

show()


#
#
#                                  mm
#                                  MM
#     `7M'   `MF'.gP"Ya   ,p6"bo mmMMmm ,pW"Wq.`7Mb,od8 ,pP"Ybd
#       VA   ,V ,M'   Yb 6M'  OO   MM  6W'   `Wb MM' "' 8I   `"
#        VA ,V  8M"""""" 8M        MM  8M     M8 MM     `YMMMa.
#         VVV   YM.    , YM.    ,  MM  YA.   ,A9 MM     L.   I8
#          W     `Mbmmd'  YMbmd'   `Mbmo`Ybmd9'.JMML.   M9mmmP'
#
#

# diagonalise matrix :
eigenvectors=eigvecs(P); # obtain eigenvector
eigentest=eigenvectors[:,ordering]; # ordered by order of growing eigenvalue
eigenvectors2=eigvecs(P2); # obtain eigenvector
eigentest2=eigenvectors2[:,ordering2]; # ordered by order of growing eigenvalu

ioff()
indexvectors=[1 2 3 4 5];
theta=0:0.001:2*pi+0.001;
fig2,axe = PyPlot.subplots(2,length(indexvectors),figsize=(20, 6))
# define scope image 


let image 
    for indexline=1:2

        if indexline==1
            eigen=eigentest
        elseif indexline==2
            eigen=eigentest2
        end

        let indexvector=1 # init, then will increm
        # for indexvector in indexvectors   
            for ax in axe[indexline,:] # the loop is over axes 
                #subplot(1,length(indexvectors),indexvector)

                ax[:set_axis_off]()
                # plot circle 
                ax.plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
                ax.set_xlim(-R-R/100,R+R/100)
                ax.set_ylim(-R-R/100,R+R/100)
                #indexvector=4; # index of the vector we want to plot 
                # Create 2-dim aray from the 1D data :
                # define scope of eigenvector 
                steps=1;
                x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
                y=zeros(1,length(x));
                transpose!(y,x);
                # initialise meshgrid 
                # initalize the amplitude map
                # ax.meshgrid(x,y)
                let eigenvector = zeros(length(x),length(x));
                    # println(indexvector)
                    # for each place point in the grid, calculate the critic value
                    for i = 1:length(x)
                        for j = 1:length(x)
                            # make sure the point is in the pool
                            if sqrt((x[i]^2+y[j]^2)) < R
                                # find closest state
                                indexneareststate=findall(sum((coordstates.-transpose(repeat([x[i] y[j]],size(coordstates,2)))).^2,dims=1)[:].==minimum(sum((coordstates.-transpose(repeat([x[i] y[j]],size(coordstates,2)))).^2,dims=1)[:]))[1];
                                # associate his coordinate on the eigenvector axis :
                                eigenvector[j,i]=eigen[indexneareststate,indexvector];
                            else
                                eigenvector[j,i] = NaN;
                            end
                        end
                    end
                    indexvector=indexvector+1; # we need to iterate the indexvector as the loop is over axe and not indexvector 
                    X,Y = np.meshgrid(x,y)
                    # global image 
                    image=ax.pcolormesh(X,Y,eigenvector,vmin=-0.06,vmax=0.04)
                end# end scope eigenvector  
                #ax.grid(False) 
                # Hide axes ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                #ax.set_title(L"$ \varphi_$(indexvector) $")
                title=latexstring("\$\\varphi_{$(indexvector-1)}\$")
                ax.set_title(title,fontweight="bold", size=20)
                # show()
            end # end loop through vectorlist
        end# end scope indexvector 
    end # end loop over the 2 lines 
# global image  # PyObject <matplotlib.collections.QuadMesh object at 0x7f8e07d74a90>
cbar=fig2[:colorbar](ax=axe,image) # adding colorbar - defined to adjust all plots to the same ref 
cbar.ax.tick_params(labelsize=16) 
# show()
savefig("Eigenvector_all_heatmap1_$(sigma)2_$(sigma2).png")
end # end scope image
close()


#
#
#       mm                     mm
#       MM                     MM
#     mmMMmm .gP"Ya  ,pP"Ybd mmMMmm
#       MM  ,M'   Yb 8I   `"   MM
#       MM  8M"""""" `YMMMa.   MM
#       MM  YM.    , L.   I8   MM
#       `Mbmo`Mbmmd' M9mmmP'   `Mbmo
#
#
ds = [1,2,3]
dc = [1.1, 1.9, 3.2]
asim = [1.5, 2.2, 3.1]
ac = [1.6, 2.15, 3.1]

categories = ["simulated", "calculated"]

p1, = plot(ds, "ko", label="D simulated")
p2, = plot(dc, "k:", label="D calculated")
p3, = plot(asim, "b+", label="A simulated")
p4, = plot(ac, "b-", label="A calculated")
p5, = plot([0], marker="None",
           linestyle="None", label="dummy-tophead")
p7, = plot([0],  marker="None",
           linestyle="None", label="dummy-empty")

leg3 = legend([p5, p1, p2, p5, p3, p4], vcat(["1"] , categories  ,["2"] , categories) ,loc=2, ncol=2) # Two columns, vertical group labels

leg4 = legend([p5, p7, p5, p7, p1, p2, p3, p4],vcat([r"$D_{etc}$", "", r"$A_{etc}$", ""] , categories , categories),loc=4, ncol=2) # Two columns, horizontal group labels

gca().add_artist(leg3)

#If there isn't a big empty spot on the plot, two legends:
#leg1 = legend([p1, p2], categories, title='D_etc', loc=0)
#leg2 = legend([p3, p4], categories, title='A_etc', loc=4)
#gca().add_artist(leg2) 

show()











