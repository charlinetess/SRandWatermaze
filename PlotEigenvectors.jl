# chose eigenvectors to pot 
indexvectors=[1,2,3,4,5]

theta=0:pi/50:2pi+pi/50;

using PyCall
using PyPlot
using LaTeXStrings
@pyimport matplotlib.patches as patch
@pyimport matplotlib.gridspec as grdspc
@pyimport numpy as np
axes_grid1 = pyimport("mpl_toolkits.axes_grid1") # for colorbar to be shaped to the axes 


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
			println(indexvector)
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

global image 
cbar=fig2[:colorbar](ax=axe,image)
cbar.ax.tick_params(labelsize=16) 


show()


savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Eigenvector_all_heatmap.png")

end # end scope image 


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

# fig2,axe = PyPlot.subplots(1,length(indexvectors),figsize=(100,10))#,gridspec_kw={"width_ratios":[1,1, 0.05]})
fig2, axe= PyPlot.subplots(1,length(indexvectors)+1,figsize=(55,10),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.1])) #vcat(ones(1,length(indexvectors)),0.05)))


	#,figsize=(101,10),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.00001])) #vcat(ones(1,length(indexvectors)),0.05)))


# fig2.subplots_adjust(wspace=0.01)
# define scope image 
# let image 
global image 

# eigenstate=zeros(length(indexvectors),numberofstates)

# [eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ]
eigenstate=reshape([eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ],length(indexvectors),numberofstates)
cmax=maximum(eigenstate)/255;
cmin=minimum(eigenstate)/255;
norm = plt.Normalize(cmin, cmax)
# reshape([i*j for i=1:2 for j=1:3],3,2)


for indexvector=1:length(indexvectors)

			# ax=axe[indexvector] #subplot(1,length(indexvectors)+1,indexvector)

			axe[indexvector][:set_axis_off]()
			# plot circle 
			axe[indexvector].plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
			axe[indexvector].set_xlim(-R-R/100,R+R/100)
			axe[indexvector].set_ylim(-R-R/100,R+R/100)
			image= axe[indexvector].scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=5);#,color="r") 
			# axe[indexvector].clim(cmin,cmax)
			#ax.grid(False) 
			# Hide axes ticks
			axe[indexvector].set_xticks([])
			axe[indexvector].set_yticks([])
			axe[indexvector].set_xticklabels([])
			axe[indexvector].set_yticklabels([])
			#ax.set_title(L"$ \varphi_$(indexvector) $")
			title=latexstring("\$\\varphi_{$(indexvector)}\$")
			axe[indexvector].set_title(title,fontweight="bold", size=20)
			println(indexvector)
		# colorbar()
end

# fig2.subplots_adjust(right=0.9)
# cbar_ax = fig2.add_axes([0.85, 0, .1, 1])# [left, bottom, width, height] of the new axes. All quantities are in fractions of figure width and height. 
axe[length(indexvectors)+1][:set_axis_off]()
# fig2.colorbar(image, ax=axes.ravel().tolist())
# fig2[:colorbar](ax=cbar_ax,image)
cbar=fig2[:colorbar](image,ax=axe[length(indexvectors)+1])#,cbar_ax)
cbar.ax.tick_params(labelsize=16) 

show()




savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Eigenvector_all_scatter.png")

end # end scope image 
	








# Test plotting with meshgrid 
# test=[1 2 1 ; 1 1 1 ; 1 1 1];
# using PyPlot
# using PyCall
# pcolormesh(test)
# colorbar()
# show()