using LightGraphs
using LinearAlgebra


# In this section, we will use the first approach taken by foster,morris and dayan solving the morris watermaze task, to obtian the value function over the watermaze and then reverse the computation of the value function from the successor representation to obtain the successorrepresentation in order to perform one shot navigation to the goal 


R= 100; # Radius of the circle in cm
r=5;# Radius of the platform  in cm
# parameters :
sigma=2.5; # widths of the bump to compute the probability transition, in case th eprobability transition is a exonential transform of the distances between points 
gamma=1; # discount factor


numberofneurons=500; # number of neurons 
numberofinputs=30; # number of inputs
numberofneighbours=8; # Is the probability distribution concerns only the nearest neighbours and is 0 elsewhere 

## Create points equally distant throughout the environment 
numberofstates=1000; # number of points 
freeparameter=20;

## INIT ACTIVITY :

# timestep
dt=0.001; # timestep in seconds
tau=1; # time constant of neuron in second
# define gain factor for the activity :
g=0.5; #  the slope of the activation function

q=4; # Chose the dimension of the space 

epsilon=0.05; #  the timescale on which the network activity fades.
alpha=0.05; # input strength 


delayinput=5; # how many tau do we initiate the activity 

delayend=30; # how long do we run the thing 

### parameter for the motion of the activity profile :

# Potential positions of the platform, being the reward locations :
numberofrewards=100;


# Potential Starting positions of the rat :
Xstart=[0.95,0,-0.95,0].*R; # East, North, West, South
Ystart=[0,0.95,0,-0.95].*R;







#The algorithm places n points, of which the kth point is put at distance sqrt(k-1/2) from the boundary (index begins with k=1), and with polar angle 2*pi*k/phi^2 where phi is the golden ratio. Exception: the last alpha*sqrt(n) points are placed on the outer boundary of the circle, and the polar radius of other points is scaled to account for that. This computation of the polar radius is done in the function radius.
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
    # scatter(r.*cos.(theta), r.*sin.(theta));#, marker='o', "m");
    X=r.*cos.(theta); 
    Y=r.*sin.(theta);
    return vcat(X, Y)
end

coordstates=sunflower(numberofstates,R,2); # this gives the coordinates as a numberofneurons*2 array 
# order states:
coordstates=sortslices(coordstates,dims=2,by=x->(x[2],x[1])); # this gives the coordinates as a numberofneurons*2 array , the first states having the lowest y value

# We actually don't need a graph
#graph= Graph(numberofstates)
#W=zeros(numberofstates,numberofstates);
## create the graph 
#for i=1:numberofstates
#	for j=1:numberofstates
#        # now we loop alogn the number of states
#        add_edge!(graph, i, j)
#        W[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
#	end
#end

###########################################################################
######################### Build reward vector ########################
###########################################################################
# in this example we need a lot of goal location because we need to reverse V=L*r to guess L that is very high dimensional 
indexrewards=sort(unique([rand(1:numberofstates) for k=1:numberofrewards]));
numberofrewards=length(indexrewards); # update the new number of neurons 

rewardscoordinates=coordstates[:,indexrewards];



# buid the matrix of rewards : basically put a 1 in the 

rewards=zeros(numberofstates,numberofrewards); # we will use this matrix to then solve the equation V=L*RemoteRef()


for j=1:numberofrewards
    for i=1:numberofstates
        if sum((coordstates[:,i].-rewardscoordinates[:,j]).^2,dims=1)[1]<r^2
            rewards[i,j]=1;
        end
    end 
end




























# we may plot the graph :  # apparently does not work on julia 1.0
#using GraphPlot
#gplot(graph)
#draw("mygraph.png", gplot(graph))

#################################################################################
########################### Build transition matrix ###############################
#################################################################################

# first trial we consider an exponential transformation of the distances :
P=zeros(numberofstates,numberofstates);
for i=1:numberofstates
   for j=1:numberofstates

        P[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
   end
        P[i,:]=exp.(-P[i,:].^2/(2*sigma^2))./sum(exp.(-P[i,:].^2/(2*sigma^2))); # normalize it to obtain a probability distribution
end

# Actually adaptation is not that easy : eigenvalues are not inferior to 1, ...  
# other option we consider a uniform probability to end up to the neighbours: 
#P=zeros(numberofstates,numberofstates);
#distances=zeros(numberofstates,numberofstates);
#for i=1:numberofstates
#    # find nearest neighbours:
#    # compute distances 
#    for j=1:numberofstates
#        distances[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2))
#    end 
#        # order the distance array from the lowest to highest value:
#        distancesordered=sort(distances[i,:],rev=false)
#        # keep only the 5 lowest values :
#        distancesselected=distancesordered[1:numberofneighbours];
#        # find corresponding neighbours :
#        indexselectedstates=[findall(distances[i,:].==distancesselected[k]) for k=1:length(distancesselected)];
#
#        # create probability distribution between states 
#        for k=1:numberofneighbours
#            P[i,indexselectedstates[k][1]]=1/numberofneighbours;
#        end
#end
#
#P=Symmetric(P)


eigenvalues=eigvals(P);
eigenvaluesbis=sort(eigenvalues,  rev=true);

# order eigenvectors
ordering=[findall(x->eigenvaluesbis[k]==x,eigenvalues)[1] for k in 1:numberofstates]; # 

# check that we got this right :
if !(eigenvaluesbis==eigenvalues[ordering])
    println("something is wrong with the order")
    #break
end


# diagonalise matrix :
eigenvectors=eigvecs(P); # obtain eigenvector
eigentest=eigenvectors[:,ordering]; # ordered by order of growing eigenvalue

indexinputs=sort(unique([rand(1:numberofstates) for k=1:numberofinputs]));
numberofinputs=length(indexinputs);


# activation function
function activation(g,x)
    return g*max(0.0,x)
end


scalevalue=ones(length(eigenvaluesbis[2:end]))./sqrt.((ones(length(eigenvaluesbis[2:end]))-gamma.*eigenvaluesbis[2:end])); # scaling factor to create the new coordinates, exclude the first one that is 1 to avoid problems with the computation

########### Play on the free parameter 
#freeparameter=sqrt(1.2*(1-gamma*eigenvaluesbis[2])^(-1)); # testing different ranges of freeparameter
freeparameter=scalevalue[1]*1.5;

newscalevalue=vcat(freeparameter, scalevalue); # changing the first coefficient as being the free parameter to control activity 
newmatrix=Diagonal(newscalevalue); # creating a diagonal matrix with it to be able to multiply each column (=eigenvector) by the right value 
matrixpoints=eigentest*newmatrix; # obtaining new coordinates for the points 

# Now we need to normalise each column to obtain the neurons coordinates 
factornorm=Diagonal([1/norm(matrixpoints[k,:]) for k in 1:size(matrixpoints,1) ]); # taking the norms of the rows and making a diagonal out of it 
matrixneurons=factornorm*matrixpoints; # new coordinates for neurons, the norm of each vector should be 1 now 

# each row is one neuron in the order, each column are their coordinate along each eigenvectors direction
neuronsencodingvector=matrixneurons[indexneurons,1:q]; # coordinates of the neurons 


# each row is one input in the order, each column are their coordinate along each eigenvectors direction
inputsencodingvectorbis=zeros(size(matrixpoints[indexinputs,1:q],2),size(matrixpoints[indexinputs,1:q],1))
inputsencodingvector=transpose!(inputsencodingvectorbis,matrixpoints[indexinputs,1:q]);

############### initialise activities ##########
# to obtain the decoding weights, necessity to input different input randomnly scattered through the environment and then compute the resulting steady state activity 
# define vector of activities of the neurons:
global totalactivities
totalactivities=activation.(g,neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla


################################ Compute Moore-penrose pseudo inverse ################################
# The psuedo inverse compute the closest solution of Az=c in the least square sense. Our problem is DA=INPUTS , and we need to find D the weights adapted to decoding the activity
# We then will compute : tAtD=tINPUTS , and then tD=tINPUTS*pinv(transpose(A)) 

totalactivitiesbis=zeros(size(totalactivities,2),size(totalactivities,1));
totalactivitiesbis=transpose!(totalactivitiesbis,totalactivities)
pseudoinv=pinv(totalactivitiesbis,0.01);

D=(pseudoinv*inputsencodingvector')'

####################################################################################################################################
################################# INPUT ACTIVITY : We input the start position for a certain amount of time  ##########################################
####################################################################################################################################



#indexstart=rand(1:4);
#indexplatform=rand(1:length(Xplatform));
indexstart=4;
indexplatform=5;
# find index of the first input state = the nearest state of the official starting points 
indexinput1=findall((minimum(sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1))[:]);
# find index of the second input state = the nearest state of the official platform point  
indexinput2=findall((minimum(sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1))[:]);




inputencodingvector1=matrixpoints[indexinput1,1:q];
inputencodingvector2=matrixpoints[indexinput2,1:q];

global activity 
activity=zeros(numberofneurons,1)
global t
t=0;
while t<delayinput*tau
global activity
activity=activity.*(1-dt/tau).+dt/tau.*activation.(g,neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector1[:]));
global t
t=t+dt;

end # end init of activity with the first input 


####################################################################################################################################
################################# then moves the activity profile to the new input location ##########################################
####################################################################################################################################
global t
t=0;
global k
k=0;
while t<delayend*tau
    global activity
    activity=activity.*(1-dt/tau).+dt/tau.*activation.(g,neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector2[:]));
    global t
    t=t+dt;
    if isinteger(k/100) # every 5 timesteps we store the picture of the activity rofile 
        # establish the grid of points in the pool
        steps=1;
        x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
        # Create matrix activity :
        act=zeros(length(x),length(x)); 
        y=zeros(1,length(x));
        transpose!(y,x);                    
       for i = 1:length(x)
            for j = 1:length(x)
                # make sure the point is in the pool
                if sqrt((x[i]^2+y[j]^2)) < R
                        indexnearestneuron=findall(sum((neuronscoordinates.-transpose(repeat([x[i] y[j]],size(neuronscoordinates,2)))).^2,dims=1)[:].==minimum(sum((neuronscoordinates.-transpose(repeat([x[i] y[j]],size(neuronscoordinates,2)))).^2,dims=1)[:]))[1];
                        #println(counting);
                        #println(i,j);
                        #  println(intersect(find(x->x==j,convert.(Int64, neuronscoordinates)[:,2]),find(x->x==i,convert.(Int64, neuronscoordinates)[:,1])));
            
                        act[j,i]=activity[indexnearestneuron];
                else
                        act[j,i]= NaN;
                end
            end
        end 

             global x,y
        x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
        y=zeros(1,length(x));
        transpose!(y,x);   
            
            # draw a small circle around the iput location to check that activity is centered around input 
            radius=2.;
            theta=0:0.001:2*pi;
            using PyPlot
            using PyCall
            #@pyimport matplotlib.patches as patch
            ioff()
            fig2 = figure("q=$(q)_$(k)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon)")
            ax2 = PyPlot.axes(xlim = (-R,R),ylim=(-R,R))
            
            # Create a Rectangle patch
            #rect = patch.Rectangle((0,0),l,L,linewidth=1,edgecolor="r",facecolor="none")
            #
            # Add the patch to the Axes
            # Assemble everything into a LineCollection
            #plot(coordstates[indexpath,1],coordstates[indexpath,2],"g")

            pcolormesh(x,x,act)
            colorbar()
            #scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
            plot(coordstates[:,indexinput2][1].+radius.*cos.(theta),coordstates[:,indexinput2][2].+radius.*sin.(theta),linewidth=2,color="r")
            plot(coordstates[:,indexinput1][1].+radius.*cos.(theta),coordstates[:,indexinput1][2].+radius.*sin.(theta),linewidth=2,color="k")
            savefig("/Users/pmxct2/Documents/FMD-Gerstner/Activity_$(q)_$(k)_$(g)_$(alpha)_$(freeparameter)_$(epsilon)_$(sigma).png")
            clf()
            close()
    end
    global k 
    k=k+1;
end # end evolution to the new goal 














