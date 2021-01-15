
# chose eigenvectors to pot 
indexvectors=[1,2,3,4,5]

theta=0:pi/50:2pi+pi/50;

using PyCall
using PyPlot
using LaTeXStrings
@pyimport matplotlib.patches as patch
@pyimport numpy as np


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

fig2,axe = PyPlot.subplots(1,length(indexvectors),figsize=(25, 6))
# define scope image 
# let image 
global image 

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

			indexvector=indexvector+1;
			X,Y = np.meshgrid(x,y)
			global image 
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

global image  # PyObject <matplotlib.collections.QuadMesh object at 0x7f8e07d74a90>
cbar=fig2[:colorbar](ax=axe,image)
cbar.ax.tick_params(labelsize=16) 
# show()


savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Eigenvector_all_heatmap.png")

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
fig2, axe= PyPlot.subplots(1,length(indexvectors)+1,figsize=(51,10),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.00000001])) #vcat(ones(1,length(indexvectors)),0.05)))
# let image 
global image 

# eigenstate=zeros(length(indexvectors),numberofstates)

# [eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ]
eigenstate=reshape([eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ],length(indexvectors),numberofstates)
global cmax, cmin
cmax=maximum(eigenstate)/255;
cmin=minimum(eigenstate)/255;
# norm = plt.Normalize(cmin, cmax)
# reshape([i*j for i=1:2 for j=1:3],3,2)


for indexvector=1:length(indexvectors)

			ax=subplot(1,length(indexvectors)+1,indexvector)

			ax[:set_axis_off]()
			# plot circle 
			ax.plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
			ax.set_xlim(-R-R/100,R+R/100)
			ax.set_ylim(-R-R/100,R+R/100)
			global image 
			image=ax.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=5);#,color="r") # this works 
			image=PyPlot.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=5);#,color="r") 

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
PyPlot.colorbar(image,ax=ax) # this works but the colorbar is very far away 


savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Eigenvector_all_scatter.png")

close()
# show()


# # colorbar takes over the last plot 
# fig2,axe = PyPlot.subplots(1,length(indexvectors),figsize=(25, 6))
# # define scope image 
# # let image 
# global image 
# # eigenstate=zeros(length(indexvectors),numberofstates)
# # [eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ]
# eigenstate=reshape([eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ],length(indexvectors),numberofstates)
# cmax=maximum(eigenstate)/255;
# cmin=minimum(eigenstate)/255;
# # reshape([i*j for i=1:2 for j=1:3],3,2)
# for indexvector in indexvectors 
# 		ax=subplot(1,length(indexvectors),indexvector)
# 		ax[:set_axis_off]()
# 		# plot circle 
# 		ax.plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
# 		ax.set_xlim(-R-R/100,R+R/100)
# 		ax.set_ylim(-R-R/100,R+R/100)
# 		global image
# 		image=PyPlot.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=3);#,color="r") 
# 		PyPlot.clim(cmin,cmax)
# 		#ax.grid(False) 
# 		# Hide axes ticks
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		ax.set_xticklabels([])
# 		ax.set_yticklabels([])
# 		#ax.set_title(L"$ \varphi_$(indexvector) $")
# 		title=latexstring("\$\\varphi_{$(indexvector-1)}\$")
# 		ax.set_title(title,fontweight="bold", size=20)
# 	# colorbar()

# end# end scope indexvector 


# cbar=PyPlot.colorbar()

# # # global image # PyObject <matplotlib.collections.PathCollection object at 0x7f8e14082ee0>
# # # cbar=fig2[:colorbar](ax=axe,image)
# # cbar.ax.tick_params(labelsize=16) 
# # cbar=fig2[:colorbar](ax=axe,image)
# # PyPlot.colorbar(image)
# cbar.ax.tick_params(labelsize=16) 
# show()






# savefig("Eigenvector_all_scatter.png")

# # end # end scope image 
	









# scalarneurons=[neuronsencodingvector[indexneuroncenter,:]'*D[:,i] for i=1:numberofneurons]



# # draw a small circle around the iput location to check that activity is centered around input 
# radiuscircle=2.;


# theta=0:0.001:2*pi;

# using PyPlot
# using PyCall



# fig2 = figure("Line Collection Example")
# ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))

# # plot circle 
# plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
# plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")

# scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=scalarneurons./255);#,color="r") # add plot of the input 

# colorbar()
# ax[:set_axis_off]()
# savefig("Eigenvectortest$(indexneuroncenter).png")


# show()



#This works fine
# Test plotting with meshgrid 
# test=[1 2 1 ; 1 1 1 ; 1 1 1];
# using PyPlot
# using PyCall
# pcolormesh(test)
# colorbar()
# show()