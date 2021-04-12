using LightGraphs
using LinearAlgebra
using PyPlot
using Colors
using PyCall
using LaTeXStrings
using FileIO 
using JLD2
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
sigmarange=2.5;
qrange= [2 4 40];
freeparamrange=[1 2 3]; # option 1 depends on the value of the other eigenvalues, option 2 doesnt 

# sigmarange=[1];
# qrange= [4];
# freeparamrange=[1]; # option 1 depends on the value of the other eigenvalues, option 2 doesnt 
# # to debug
# sigma=sigmarange[1];
# q=qrange[1];
# freeparam=freeparamrange[1];



# sigma=5 # up to 13 the matrix is still diagonalisable 





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
indexvectors=[1 2 3 4 5];
# find index of the first input state = the nearest state of the official starting points 
indexinput1=findall((minimum(sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1))[:]);
# find index of the second input state = the nearest state of the official platform point  
indexinput2=findall((minimum(sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1))[:]);

# or we can define them according to their distance to the center, but then hard to know if they will be at similar angles or not 
# indexinput1=45;
# indexinput2=900;

for sigma in sigmarange
	for q in qrange 
		for freeparam in freeparamrange 
			let R,r,gamma,numberofneurons,numberofstates,coordstates,dt,tau,g,epsilon,alpha,delayinput,delayend,Xplatform,Yplatform,Xstart,Ystart, indexstart, indexplatform, indexinput1, indexinput2, P, eigenvalues,inputsencodingvector, matrixneurons, neuronsencodingvector, eigentest, indexneurons, numberofneurons, neuronscoordinates, numberofinputs, scalevalue, D, matrixpoints, activity, epsilon, alpha, delayinput, delayend, x, y, activity, trajectory, t, k, neuronsencodingvector, theta=0:0.001:2*pi+0.001; # define scope 
			

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
			# 	 .gP"Ya `7MMpMMMb.`7M'   `MF'
			# 	,M'   Yb  MM    MM  VA   ,V
			# 	8M""""""  MM    MM   VA ,V
			# 	YM.    ,  MM    MM    VVV
			# 	 `Mbmmd'.JMML  JMML.   W
			# 	
			# 	

			# first trial we consider an exponential transformation of the distances :
			P=zeros(numberofstates,numberofstates);
			for i=1:numberofstates
			   for j=1:numberofstates
			        P[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
			   end
			        P[i,:]=exp.(-P[i,:].^2/(2*sigma^2))./sum(exp.(-P[i,:].^2/(2*sigma^2))); # normalize it to obtain a probability distribution
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

			# diagonalise matrix :
			eigenvectors=eigvecs(P); # obtain eigenvector
			eigentest=eigenvectors[:,ordering]; # ordered by order of growing eigenvalue

			# define neurons 
			indexneurons=sort(unique([rand(1:numberofstates) for k=1:numberofneurons]));
			numberofneurons=length(indexneurons); # update the new number of neurons 

			neuronscoordinates=coordstates[:,indexneurons];

			indexinputs=sort(unique([rand(1:numberofstates) for k=1:numberofinputs]));
			numberofinputs=length(indexinputs);

			scalevalue=ones(length(eigenvaluesbis[2:end]))./sqrt.((ones(length(eigenvaluesbis[2:end]))-gamma.*eigenvaluesbis[2:end])); # scaling factor to create the new coordinates, exclude the first one that is 1 to avoid problems with the computation
			########### Play on the free parameter                                   .JMML.
			if freeparam==1 # if option 1 we use 
				freeparameter=sqrt(1.2*(1-gamma*eigenvaluesbis[2])^(-1)); # this is usually quite high 
				# freeparameter=scalevalue[1]*1.5;
			elseif freeparam==2 # if option 2 we use 
				freeparameter=0.001; # this is usually quite low 
			elseif freeparam==3;
				freeparameter=3*sqrt((1-gamma*eigenvaluesbis[2])^(-1)); # exactly equal to the second component  
			end 

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
			# 	
			# 	                           ,,            ,,
			# 	                           db          `7MM        mm
			# 	                                         MM        MM
			# 	`7M'    ,A    `MF'.gP"Ya `7MM  .P"Ybmmm  MMpMMMb.mmMMmm ,pP"Ybd
			# 	  VA   ,VAA   ,V ,M'   Yb  MM :MI  I8    MM    MM  MM   8I   `"
			# 	   VA ,V  VA ,V  8M""""""  MM  WmmmP"    MM    MM  MM   `YMMMa.
			# 	    VVV    VVV   YM.    ,  MM 8M         MM    MM  MM   L.   I8
			# 	     W      W     `Mbmmd'.JMML.YMMMMMb .JMML  JMML.`MbmoM9mmmP'
			# 	                              6'     dP
			# 	                              Ybmmmd'

			# to obtain the decoding weights, necessity to input different input randomnly scattered through the environment and then compute the resulting steady state activity 
			# define vector of activities of the neurons:
			# global totalactivities
			totalactivities=activation.(g,neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla
			# totalactivities=complexactivation.(neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla

			################################ Compute Moore-penrose pseudo inverse ################################
			# The psuedo inverse compute the closest solution of Az=c in the least square sense. Our problem is DA=INPUTS , and we need to find D the weights adapted to decoding the activity
			# We then will compute : tAtD=tINPUTS , and then tD=tINPUTS*pinv(transpose(A)) 

			totalactivitiesbis=zeros(size(totalactivities,2),size(totalactivities,1));
			totalactivitiesbis=transpose!(totalactivitiesbis,totalactivities)
			pseudoinv=pinv(totalactivitiesbis,0.01);

			D=(pseudoinv*inputsencodingvector')'


			# 	
			# 	           ,,                                          ,,          ,...       ,,        ,,
			# 	`7MM"""YMM db                                        `7MM        .d' ""     `7MM      `7MM
			# 	  MM    `7                                             MM        dM`          MM        MM
			# 	  MM   d `7MM  .P"Ybmmm      ,6"Yb.  `7MMpMMMb.   ,M""bMM       mMMmm,pW"Wq.  MM   ,M""bMM  .gP"Ya `7Mb,od8
			# 	  MM""MM   MM :MI  I8       8)   MM    MM    MM ,AP    MM        MM 6W'   `Wb MM ,AP    MM ,M'   Yb  MM' "'
			# 	  MM   Y   MM  WmmmP"        ,pm9MM    MM    MM 8MI    MM        MM 8M     M8 MM 8MI    MM 8M""""""  MM
			# 	  MM       MM 8M            8M   MM    MM    MM `Mb    MM        MM YA.   ,A9 MM `Mb    MM YM.    ,  MM
			# 	.JMML.   .JMML.YMMMMMb      `Moo9^Yo..JMML  JMML.`Wbmd"MML.    .JMML.`Ybmd9'.JMML.`Wbmd"MML.`Mbmmd'.JMML.
			# 	              6'     dP
			# 	              Ybmmmd'
			if !isdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)")
				mkdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)") # create the new repo only if doesnt exist 
			end

			# 	
			# 	           ,,                              ,,
			# 	         `7MM           mm                 db
			# 	           MM           MM
			# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm      .gP"Ya `7MM  .P"Ybmmm .gP"Ya `7MMpMMMb.
			# 	  MM   `Wb MM 6W'   `Wb MM       ,M'   Yb  MM :MI  I8  ,M'   Yb  MM    MM
			# 	  MM    M8 MM 8M     M8 MM       8M""""""  MM  WmmmP"  8M""""""  MM    MM
			# 	  MM   ,AP MM YA.   ,A9 MM       YM.    ,  MM 8M       YM.    ,  MM    MM
			# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo     `Mbmmd'.JMML.YMMMMMb  `Mbmmd'.JMML  JMML.
			# 	  MM                                          6'     dP
			# 	.JMML.                                        Ybmmmd'
			# include("PlotEigenvectors.jl") # this generates 2 figure one heatmap and one scatterplot 



			# 	
			# 	  ,,
			# 	`7MM                         mm
			# 	  MM                         MM
			# 	  MMpMMMb.  .gP"Ya   ,6"Yb.mmMMmm `7MMpMMMb.pMMMb.   ,6"Yb. `7MMpdMAo.
			# 	  MM    MM ,M'   Yb 8)   MM  MM     MM    MM    MM  8)   MM   MM   `Wb
			# 	  MM    MM 8M""""""  ,pm9MM  MM     MM    MM    MM   ,pm9MM   MM    M8
			# 	  MM    MM YM.    , 8M   MM  MM     MM    MM    MM  8M   MM   MM   ,AP
			# 	.JMML  JMML.`Mbmmd' `Moo9^Yo.`Mbmo.JMML  JMML  JMML.`Moo9^Yo. MMbmmd'
			# 	                                                              MM
			# 	                                                            .JMML.
			ioff()

			fig2,axe = PyPlot.subplots(1,length(indexvectors),figsize=(20, 3))
			# define scope image 
			let image 
				let indexvector=1 # init, then will increm
				# for indexvector in indexvectors	
					for ax in axe # the loop is over axes 
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
										eigenvector[j,i]=eigentest[indexneareststate,indexvector];
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
			# global image  # PyObject <matplotlib.collections.QuadMesh object at 0x7f8e07d74a90>
			cbar=fig2[:colorbar](ax=axe,image) # adding colorbar - defined to adjust all plots to the same ref 
			cbar.ax.tick_params(labelsize=16) 
			# show()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/Eigenvector_all_heatmap.png")
			end # end scope image
			close()

			# 	
			# 	
			# 	                          mm     mm
			# 	                          MM     MM
			# 	,pP"Ybd  ,p6"bo   ,6"Yb.mmMMmm mmMMmm .gP"Ya `7Mb,od8
			# 	8I   `" 6M'  OO  8)   MM  MM     MM  ,M'   Yb  MM' "'
			# 	`YMMMa. 8M        ,pm9MM  MM     MM  8M""""""  MM
			# 	L.   I8 YM.    , 8M   MM  MM     MM  YM.    ,  MM
			# 	M9mmmP'  YMbmd'  `Moo9^Yo.`Mbmo  `Mbmo`Mbmmd'.JMML.
			# 	
			ioff()

			# colorbar has its own place but a bit far from the plot - still prefered solution 
			fig2, axe= PyPlot.subplots(1,length(indexvectors)+1,figsize=(20,3),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.00000001])) #vcat(ones(1,length(indexvectors)),0.05)))
			# let image 
			let image, cmin, cmax
				# eigenstate=zeros(length(indexvectors),numberofstates)
				# [eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ]
				eigenstate=reshape([eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ],length(indexvectors),numberofstates)
				# global cmax, cmin
				cmax=maximum(eigenstate);
				cmin=minimum(eigenstate);
				# norm = plt.Normalize(cmin, cmax)
				# reshape([i*j for i=1:2 for j=1:3],3,2)
					for indexvector=1:length(indexvectors)
						ax=subplot(1,length(indexvectors)+1,indexvector)
						ax[:set_axis_off]()
						# plot circle 
						ax.plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
						ax.set_xlim(-R-R/100,R+R/100)
						ax.set_ylim(-R-R/100,R+R/100)
						# global image 
						# image=ax.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=8);#,color="r") # this works 
						image=PyPlot.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:],s=12);#,color="r") 
						PyPlot.clim(cmin,cmax)
						# quadmesh.set_clim(cmin,cmax)
						#ax.grid(False) 
						# Hide axes ticks
						ax.set_xticks([])
						ax.set_yticks([])
						ax.set_xticklabels([])
						ax.set_yticklabels([])
						#ax.set_title(L"$ \varphi_$(indexvector) $")
						title=latexstring("\$\\varphi_{$(indexvector-1)}\$")
						ax.set_title(title,fontweight="bold", size=20)
					end
				ax=subplot(1,length(indexvectors)+1,length(indexvectors)+1)
				ax[:set_axis_off]()
				ax.set_xticks([])
				ax.set_yticks([])
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				# fig2.subplots_adjust(wspace=0.01)
				cbar=PyPlot.colorbar(image,ax=ax) # this works but the colorbar is very far away 
				cbar.ax.tick_params(labelsize=16) 
				savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/Eigenvector_all_scatter.png")
			end # end scope image, cmin, cmax 
			close()
			# show()



			# 	
			# 	           ,,                      ,,
			# 	         `7MM           mm         db                                    mm
			# 	           MM           MM                                               MM
			# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm     `7MM  `7MMpMMMb. `7MMpdMAo.`7MM  `7MM mmMMmm
			# 	  MM   `Wb MM 6W'   `Wb MM         MM    MM    MM   MM   `Wb  MM    MM   MM
			# 	  MM    M8 MM 8M     M8 MM         MM    MM    MM   MM    M8  MM    MM   MM
			# 	  MM   ,AP MM YA.   ,A9 MM         MM    MM    MM   MM   ,AP  MM    MM   MM
			# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    .JMML..JMML  JMML. MMbmmd'   `Mbod"YML. `Mbmo
			# 	  MM                                                MM
			# 	.JMML.                                            .JMML.
			# include("plotInputStrength.jl") # this generates 3 figures for 3 different neurons at different locations

			# this plot Fig 7.3 of the thesis : it shows the strength of the input for input scattered around the maze to the neuron defined by "indexneuroncenter" 

			ioff()
			# plot the scalar product between 2 neurons of the space 
			indexneuroncenter=20; # index of the circle location 
			# establish the grid of points in the pool
			# draw a small circle around the iput location to check that activity is centered around input 
			radiuscircle=2.;
			theta=0:0.001:2*pi+0.001;
			scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
			fig2 = figure("Line Collection Example")
			ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
			# plot circle 
			plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
			plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
			scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate);#,color="r") # add plot of the input 

			colorbar()
			ax[:set_axis_off]()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/Input$(indexneuroncenter).png")
			close()
			ioff()

			indexneuroncenter=100; # index of the circle location 
			# establish the grid of points in the pool
			# draw a small circle around the iput location to check that activity is centered around input 
			radiuscircle=2.;
			theta=0:0.001:2*pi+0.001;
			scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
			fig2 = figure("Line Collection Example")
			ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
			# plot circle 
			plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
			plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
			scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate);#,color="r") # add plot of the input 

			colorbar()
			ax[:set_axis_off]()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/Input$(indexneuroncenter).png")
			close()
			ioff()

			indexneuroncenter=300; # index of the circle location 
			# establish the grid of points in the pool
			# draw a small circle around the iput location to check that activity is centered around input 
			radiuscircle=2.;
			theta=0:0.001:2*pi+0.001;
			scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
			fig2 = figure("Line Collection Example")
			ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
			# plot circle 
			plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
			plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
			scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate);#,color="r") # add plot of the input 

			colorbar()
			ax[:set_axis_off]()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/Input$(indexneuroncenter).png")
			close()
			# 	           ,,
			# 	         `7MM           mm       `7MM"""Mq.
			# 	           MM           MM         MM   `MM.
			# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm       MM   ,M9  .gP"Ya   ,p6"bo
			# 	  MM   `Wb MM 6W'   `Wb MM         MMmmdM9  ,M'   Yb 6M'  OO
			# 	  MM    M8 MM 8M     M8 MM         MM  YM.  8M"""""" 8M
			# 	  MM   ,AP MM YA.   ,A9 MM         MM   `Mb.YM.    , YM.    ,
			# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    .JMML. .JMM.`Mbmmd'  YMbmd'
			# 	  MM
			# 	.JMML.
			# include("plotRecurrentWeights.jl") # this generates 3 figures for 3 different neurons at different locations


			ioff()
			indexneuroncenter=100; # index of the circle location 
			scalarneurons=[neuronsencodingvector[indexneuroncenter,:]'*D[:,i] for i=1:numberofneurons];
			# draw a small circle around the iput location to check that activity is centered around input 
			radiuscircle=2.;
			theta=0:0.001:2*pi;
			fig2 = figure("Line Collection Example")
			ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
			# plot circle 
			plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
			plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
			scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=scalarneurons);#,color="r") # add plot of the input 
			colorbar()
			ax[:set_axis_off]()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/RecurrentWeights$(indexneuroncenter).png")
			close()



			ioff()
			indexneuroncenter=20; # index of the circle location 
			scalarneurons=[neuronsencodingvector[indexneuroncenter,:]'*D[:,i] for i=1:numberofneurons]
			# draw a small circle around the iput location to check that activity is centered around input 
			radiuscircle=2.;
			theta=0:0.001:2*pi;
			fig2 = figure("Line Collection Example")
			ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
			# plot circle 
			plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
			plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
			scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=scalarneurons);#,color="r") # add plot of the input 
			colorbar()
			ax[:set_axis_off]()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/RecurrentWeights$(indexneuroncenter).png")
			close()


			ioff()
			indexneuroncenter=300; # index of the circle location 
			scalarneurons=[neuronsencodingvector[indexneuroncenter,:]'*D[:,i] for i=1:numberofneurons]
			# draw a small circle around the iput location to check that activity is centered around input 
			radiuscircle=2.;
			theta=0:0.001:2*pi;
			fig2 = figure("Line Collection Example")
			ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
			# plot circle 
			plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
			plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
			scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=scalarneurons);#,color="r") # add plot of the input 
			colorbar()
			ax[:set_axis_off]()
			savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/RecurrentWeights$(indexneuroncenter).png")
			close()

			# 	
			# 	  ,,                ,,
			# 	  db                db   mm
			# 	                         MM
			# 	`7MM  `7MMpMMMb.  `7MM mmMMmm
			# 	  MM    MM    MM    MM   MM
			# 	  MM    MM    MM    MM   MM
			# 	  MM    MM    MM    MM   MM
			# 	.JMML..JMML  JMML..JMML. `Mbmo
			# 	
			# 	
			inputencodingvector1=matrixpoints[indexinput1,1:q];
			inputencodingvector2=matrixpoints[indexinput2,1:q];
			# define trajectory of the maximum 
			# global trajectory
			trajectory=[];
			# input the start location: 
			# global activity 
			activity=zeros(numberofneurons,1)
			# global t
			t=0;
			while t<delayinput*tau
				# global activity
				activity=activity.*(1-dt/tau).+dt/tau.*activation.(g,neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector1[:]));
				# activity=activity.*(1-dt/tau).+dt/tau.*complexactivation.(neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector1[:]));
				# global trajectory
				trajectory=push!(trajectory,findall(maximum(activity).==activity)[1]);
				# global t
				t=t+dt;
			end # end init of activity with the first input 

			# # 	
			# # 	                                                                            ,,
			# # 	`7MMM.     ,MMF'                                                          `7MM
			# # 	  MMMb    dPMM                                                              MM
			# # 	  M YM   ,M MM  ,pW"Wq.`7M'   `MF'.gP"Ya       .P"Ybmmm ,pW"Wq.   ,6"Yb.    MM
			# # 	  M  Mb  M' MM 6W'   `Wb VA   ,V ,M'   Yb     :MI  I8  6W'   `Wb 8)   MM    MM
			# # 	  M  YM.P'  MM 8M     M8  VA ,V  8M""""""      WmmmP"  8M     M8  ,pm9MM    MM
			# # 	  M  `YM'   MM YA.   ,A9   VVV   YM.    ,     8M       YA.   ,A9 8M   MM    MM
			# # 	.JML. `'  .JMML.`Ybmd9'     W     `Mbmmd'      YMMMMMb  `Ybmd9'  `Moo9^Yo..JMML.
			# # 	                                              6'     dP
			# # 	           
			# create directories                               
			if !isdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/ActivitiesMesh")
				mkdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/ActivitiesMesh") # create the new repo only if doesnt exist 
			end
			if !isdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/ActivitiesScatter")
				mkdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/ActivitiesScatter") # create the new repo only if doesnt exist 
			end
			if !isdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/MaxActivity")
				mkdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/MaxActivity") # create the new repo only if doesnt exist 
			end
			if !isdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/RecurrentInput")
				mkdir("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/RecurrentInput") # create the new repo only if doesnt exist 
			end

			# global t
			t=0;
			# global k
			k=0;
				while t<delayend*tau
				    # global activity
				        activity=activity.*(1-dt/tau).+dt/tau.*activation.(g,neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector2[:]));
				        # activity=activity.*(1-dt/tau).+dt/tau.*complexactivation.(neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector2[:]));
				    # global t
				    t=t+dt;
				    # global trajectory
				    trajectory=push!(trajectory,findall(maximum(activity).==activity)[1]);
				    if isinteger(k/10) # every 5 timesteps we store the picture of the activity rofile 
				        # 	
				        # 	           ,,                                                       ,,
				        # 	         `7MM           mm                                        `7MM
				        # 	           MM           MM                                          MM
				        # 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm     `7MMpMMMb.pMMMb.  .gP"Ya  ,pP"Ybd  MMpMMMb.
				        # 	  MM   `Wb MM 6W'   `Wb MM         MM    MM    MM ,M'   Yb 8I   `"  MM    MM
				        # 	  MM    M8 MM 8M     M8 MM         MM    MM    MM 8M"""""" `YMMMa.  MM    MM
				        # 	  MM   ,AP MM YA.   ,A9 MM         MM    MM    MM YM.    , L.   I8  MM    MM
				        # 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    .JMML  JMML  JMML.`Mbmmd' M9mmmP'.JMML  JMML.
				        # 	  MM
				        # 	.JMML.
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
				        # global x,y
				        x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
				        y=zeros(1,length(x));
				        transpose!(y,x);   
				            # draw a small circle around the iput location to check that activity is centered around input 
				            radiuscircle=2.;
				            theta=0:0.001:2*pi+0.001;
				            ioff()
				            fig2 = figure("q=$(q)_$(k)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon)")
				            ax2 = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
				            pcolormesh(x,x,act)
				            colorbar()
				            ax2[:grid](false);
				            ax2[:set_axis_off]();
				            #scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
				            plot(coordstates[:,indexinput2][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput2][2].+radiuscircle.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
				            plot(coordstates[:,indexinput1][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput1][2].+radiuscircle.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
				            plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
				            savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/ActivitiesMesh/Activity_$(k).png")
				            close()

				            # 	
				            # 	           ,,
				            # 	         `7MM           mm                                 mm     mm
				            # 	           MM           MM                                 MM     MM
				            # 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm     ,pP"Ybd  ,p6"bo   ,6"Yb.mmMMmm mmMMmm .gP"Ya `7Mb,od8
				            # 	  MM   `Wb MM 6W'   `Wb MM       8I   `" 6M'  OO  8)   MM  MM     MM  ,M'   Yb  MM' "'
				            # 	  MM    M8 MM 8M     M8 MM       `YMMMa. 8M        ,pm9MM  MM     MM  8M""""""  MM
				            # 	  MM   ,AP MM YA.   ,A9 MM       L.   I8 YM.    , 8M   MM  MM     MM  YM.    ,  MM
				            # 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    M9mmmP'  YMbmd'  `Moo9^Yo.`Mbmo  `Mbmo`Mbmmd'.JMML.
				            # 	  MM
				            # 	.JMML.
						 #    ioff()
				   #          fig2 = figure("q=$(q)_$(k)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon)")
				   #          ax2 = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
							# scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=activity);#,color="r") # add plot of the input 
							# ax2[:set_axis_off]()
				   #          colorbar()
				   #          ax2[:grid](false);
				   #          plot(coordstates[:,indexinput2][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput2][2].+radiuscircle.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
				   #          plot(coordstates[:,indexinput1][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput1][2].+radiuscircle.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
				   #          plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
				   #          savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/ActivitiesScatter/Activity_$(k).png")
				   #          clf()
				   #          close()

				            # 	
				            # 	           ,,
				            # 	         `7MM           mm
				            # 	           MM           MM
				            # 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm     `7MMpMMMb.pMMMb.   ,6"Yb.  `7M'   `MF'
				            # 	  MM   `Wb MM 6W'   `Wb MM         MM    MM    MM  8)   MM    `VA ,V'
				            # 	  MM    M8 MM 8M     M8 MM         MM    MM    MM   ,pm9MM      XMX
				            # 	  MM   ,AP MM YA.   ,A9 MM         MM    MM    MM  8M   MM    ,V' VA.
				            # 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    .JMML  JMML  JMML.`Moo9^Yo..AM.   .MA.
				            # 	  MM
				            # 	.JMML.
						 #    ioff()
				   #          fig2 = figure("q=$(q)_$(k)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon)")
				   #          ax2 = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
				   #          colorMaxAct=zeros(numberofneurons,1)
				   #          colorMaxAct[findall(x->x==maximum(activity),activity)]=colorMaxAct[findall(x->x==maximum(activity),activity)].+0.8;

							# scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=colorMaxAct,vmin=0.,vmax=1.,cmap="jet");#,color="r") # add plot of the input 
							# ax2[:set_axis_off]()
				   #          ax2[:grid](false);
				   #          plot(coordstates[:,indexinput2][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput2][2].+radiuscircle.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
				   #          plot(coordstates[:,indexinput1][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput1][2].+radiuscircle.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
				   #          plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
				   #          savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/MaxActivity/Activity_$(k).png")
				   #          clf()
				   #          close()

							# 	
							# 	           ,,
							# 	         `7MM           mm       `7MM"""Mq.                 `7MMF'                                  mm
							# 	           MM           MM         MM   `MM.                  MM                                    MM
							# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm       MM   ,M9  .gP"Ya   ,p6"bo  MM  `7MMpMMMb. `7MMpdMAo.`7MM  `7MM mmMMmm
							# 	  MM   `Wb MM 6W'   `Wb MM         MMmmdM9  ,M'   Yb 6M'  OO  MM    MM    MM   MM   `Wb  MM    MM   MM
							# 	  MM    M8 MM 8M     M8 MM         MM  YM.  8M"""""" 8M       MM    MM    MM   MM    M8  MM    MM   MM
							# 	  MM   ,AP MM YA.   ,A9 MM         MM   `Mb.YM.    , YM.    , MM    MM    MM   MM   ,AP  MM    MM   MM
							# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    .JMML. .JMM.`Mbmmd'  YMbmd'.JMML..JMML  JMML. MMbmmd'   `Mbod"YML. `Mbmo
							# 	  MM                                                                           MM
							# 	.JMML.      

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
				                        act[j,i]=(neuronsencodingvector*D*activity)[indexnearestneuron];
				                else 
				                        act[j,i]= NaN;
				                end
				            end
				        end 
				        # global x,y
				        x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
				        y=zeros(1,length(x));
				        transpose!(y,x);   
				            # draw a small circle around the iput location to check that activity is centered around input 
				            radiuscircle=2.;
				            theta=0:0.001:2*pi+0.001;
				            ioff()
				            fig2 = figure("q=$(q)_$(k)_g=$(g)_alpha=$(alpha)_freeparam=$(freeparameter)epsilon=_$(epsilon)")
				            ax2 = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
				            pcolormesh(x,x,act)
				            colorbar()
				            ax2[:grid](false);
				            ax2[:set_axis_off]();
				            #scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
				            plot(coordstates[:,indexinput2][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput2][2].+radiuscircle.*sin.(theta),linewidth=2,color=[250/255,128/255,114/255])
				            plot(coordstates[:,indexinput1][1].+radiuscircle.*cos.(theta),coordstates[:,indexinput1][2].+radiuscircle.*sin.(theta),linewidth=2,color=[169/255,169/255,169/255])
				            plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
				            savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparameter)/RecurrentInput/Activity_$(k).png")
				            close()                                                                

				    end
				    # global k 
				    k=k+1;
				end # end evolution to the new goal 

			end # end scope 
		end # end freeparam
	end # end q
end # end sigma




