ï
Ó
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

û
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8¦Ë
¶
/Adam/residual_unit/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/residual_unit/batch_normalization_2/beta/v
¯
CAdam/residual_unit/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp/Adam/residual_unit/batch_normalization_2/beta/v*
_output_shapes
:
*
dtype0
¸
0Adam/residual_unit/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20Adam/residual_unit/batch_normalization_2/gamma/v
±
DAdam/residual_unit/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp0Adam/residual_unit/batch_normalization_2/gamma/v*
_output_shapes
:
*
dtype0
¬
$Adam/residual_unit/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*5
shared_name&$Adam/residual_unit/conv2d_2/kernel/v
¥
8Adam/residual_unit/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/residual_unit/conv2d_2/kernel/v*&
_output_shapes
:

*
dtype0
¶
/Adam/residual_unit/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/residual_unit/batch_normalization_1/beta/v
¯
CAdam/residual_unit/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp/Adam/residual_unit/batch_normalization_1/beta/v*
_output_shapes
:
*
dtype0
¸
0Adam/residual_unit/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20Adam/residual_unit/batch_normalization_1/gamma/v
±
DAdam/residual_unit/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp0Adam/residual_unit/batch_normalization_1/gamma/v*
_output_shapes
:
*
dtype0
¬
$Adam/residual_unit/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*5
shared_name&$Adam/residual_unit/conv2d_1/kernel/v
¥
8Adam/residual_unit/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/residual_unit/conv2d_1/kernel/v*&
_output_shapes
:

*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:

*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:
*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:
*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:
*
dtype0
¶
/Adam/residual_unit/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/residual_unit/batch_normalization_2/beta/m
¯
CAdam/residual_unit/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp/Adam/residual_unit/batch_normalization_2/beta/m*
_output_shapes
:
*
dtype0
¸
0Adam/residual_unit/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20Adam/residual_unit/batch_normalization_2/gamma/m
±
DAdam/residual_unit/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp0Adam/residual_unit/batch_normalization_2/gamma/m*
_output_shapes
:
*
dtype0
¬
$Adam/residual_unit/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*5
shared_name&$Adam/residual_unit/conv2d_2/kernel/m
¥
8Adam/residual_unit/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/residual_unit/conv2d_2/kernel/m*&
_output_shapes
:

*
dtype0
¶
/Adam/residual_unit/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/residual_unit/batch_normalization_1/beta/m
¯
CAdam/residual_unit/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp/Adam/residual_unit/batch_normalization_1/beta/m*
_output_shapes
:
*
dtype0
¸
0Adam/residual_unit/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20Adam/residual_unit/batch_normalization_1/gamma/m
±
DAdam/residual_unit/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp0Adam/residual_unit/batch_normalization_1/gamma/m*
_output_shapes
:
*
dtype0
¬
$Adam/residual_unit/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*5
shared_name&$Adam/residual_unit/conv2d_1/kernel/m
¥
8Adam/residual_unit/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/residual_unit/conv2d_1/kernel/m*&
_output_shapes
:

*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:

*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:
*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:
*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
¾
3residual_unit/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53residual_unit/batch_normalization_2/moving_variance
·
Gresidual_unit/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp3residual_unit/batch_normalization_2/moving_variance*
_output_shapes
:
*
dtype0
¶
/residual_unit/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/residual_unit/batch_normalization_2/moving_mean
¯
Cresidual_unit/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp/residual_unit/batch_normalization_2/moving_mean*
_output_shapes
:
*
dtype0
¾
3residual_unit/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53residual_unit/batch_normalization_1/moving_variance
·
Gresidual_unit/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp3residual_unit/batch_normalization_1/moving_variance*
_output_shapes
:
*
dtype0
¶
/residual_unit/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/residual_unit/batch_normalization_1/moving_mean
¯
Cresidual_unit/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp/residual_unit/batch_normalization_1/moving_mean*
_output_shapes
:
*
dtype0
¨
(residual_unit/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(residual_unit/batch_normalization_2/beta
¡
<residual_unit/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp(residual_unit/batch_normalization_2/beta*
_output_shapes
:
*
dtype0
ª
)residual_unit/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)residual_unit/batch_normalization_2/gamma
£
=residual_unit/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp)residual_unit/batch_normalization_2/gamma*
_output_shapes
:
*
dtype0

residual_unit/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*.
shared_nameresidual_unit/conv2d_2/kernel

1residual_unit/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpresidual_unit/conv2d_2/kernel*&
_output_shapes
:

*
dtype0
¨
(residual_unit/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(residual_unit/batch_normalization_1/beta
¡
<residual_unit/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp(residual_unit/batch_normalization_1/beta*
_output_shapes
:
*
dtype0
ª
)residual_unit/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)residual_unit/batch_normalization_1/gamma
£
=residual_unit/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp)residual_unit/batch_normalization_1/gamma*
_output_shapes
:
*
dtype0

residual_unit/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*.
shared_nameresidual_unit/conv2d_1/kernel

1residual_unit/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpresidual_unit/conv2d_1/kernel*&
_output_shapes
:

*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:

*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:
*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:
*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:
*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:
*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:
*
dtype0

serving_default_conv2d_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
±
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceresidual_unit/conv2d_1/kernel)residual_unit/batch_normalization_1/gamma(residual_unit/batch_normalization_1/beta/residual_unit/batch_normalization_1/moving_mean3residual_unit/batch_normalization_1/moving_varianceresidual_unit/conv2d_2/kernel)residual_unit/batch_normalization_2/gamma(residual_unit/batch_normalization_2/beta/residual_unit/batch_normalization_2/moving_mean3residual_unit/batch_normalization_2/moving_variancedense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_286237

NoOpNoOp
Êq
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*q
valueûpBøp Bñp

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
¾
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 _jit_compiled_convolution_op*
Õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 axis
	!gamma
"beta
#moving_mean
$moving_variance*

%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 

+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
²
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7main_layers
8skip_layers*

9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 

?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
¦
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias*

0
!1
"2
#3
$4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
K15
L16*
R
0
!1
"2
M3
N4
O5
P6
Q7
R8
K9
L10*
* 
°
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
\trace_0
]trace_1
^trace_2
_trace_3* 
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
* 
 
diter

ebeta_1

fbeta_2
	gdecay
hlearning_ratemí!mî"mïKmðLmñMmòNmóOmôPmõQmöRm÷vø!vù"vúKvûLvüMvýNvþOvÿPvQvRv*

iserving_default* 

0*

0*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
!0
"1
#2
$3*

!0
"1*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
J
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9*
.
M0
N1
O2
P3
Q4
R5*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
$
0
1
3
4*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

¡trace_0* 

¢trace_0* 

K0
L1*

K0
L1*
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

¨trace_0* 

©trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEresidual_unit/conv2d_1/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)residual_unit/batch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(residual_unit/batch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEresidual_unit/conv2d_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)residual_unit/batch_normalization_2/gamma&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(residual_unit/batch_normalization_2/beta'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/residual_unit/batch_normalization_1/moving_mean'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3residual_unit/batch_normalization_1/moving_variance'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/residual_unit/batch_normalization_2/moving_mean'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3residual_unit/batch_normalization_2/moving_variance'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
.
#0
$1
S2
T3
U4
V5*
<
0
1
2
3
4
5
6
7*

ª0
«1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

#0
$1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
S0
T1
U2
V3*
$
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
Å
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses

Mkernel
!²_jit_compiled_convolution_op*
Ü
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses
	¹axis
	Ngamma
Obeta
Smoving_mean
Tmoving_variance*
Å
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses

Pkernel
!À_jit_compiled_convolution_op*
Ü
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses
	Çaxis
	Qgamma
Rbeta
Umoving_mean
Vmoving_variance*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
È	variables
É	keras_api

Êtotal

Ëcount*
M
Ì	variables
Í	keras_api

Îtotal

Ïcount
Ð
_fn_kwargs*

M0*

M0*
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses*
* 
* 
* 
 
N0
O1
S2
T3*

N0
O1*
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses*

Ûtrace_0
Ütrace_1* 

Ýtrace_0
Þtrace_1* 
* 

P0*

P0*
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses*
* 
* 
* 
 
Q0
R1
U2
V3*

Q0
R1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses*

étrace_0
êtrace_1* 

ëtrace_0
ìtrace_1* 
* 

Ê0
Ë1*

È	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Î0
Ï1*

Ì	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 

S0
T1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/residual_unit/conv2d_1/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/residual_unit/batch_normalization_1/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/residual_unit/batch_normalization_1/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/residual_unit/conv2d_2/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/residual_unit/batch_normalization_2/gamma/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/residual_unit/batch_normalization_2/beta/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/residual_unit/conv2d_1/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/residual_unit/batch_normalization_1/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/residual_unit/batch_normalization_1/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/residual_unit/conv2d_2/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0Adam/residual_unit/batch_normalization_2/gamma/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/Adam/residual_unit/batch_normalization_2/beta/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
î
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp1residual_unit/conv2d_1/kernel/Read/ReadVariableOp=residual_unit/batch_normalization_1/gamma/Read/ReadVariableOp<residual_unit/batch_normalization_1/beta/Read/ReadVariableOp1residual_unit/conv2d_2/kernel/Read/ReadVariableOp=residual_unit/batch_normalization_2/gamma/Read/ReadVariableOp<residual_unit/batch_normalization_2/beta/Read/ReadVariableOpCresidual_unit/batch_normalization_1/moving_mean/Read/ReadVariableOpGresidual_unit/batch_normalization_1/moving_variance/Read/ReadVariableOpCresidual_unit/batch_normalization_2/moving_mean/Read/ReadVariableOpGresidual_unit/batch_normalization_2/moving_variance/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp8Adam/residual_unit/conv2d_1/kernel/m/Read/ReadVariableOpDAdam/residual_unit/batch_normalization_1/gamma/m/Read/ReadVariableOpCAdam/residual_unit/batch_normalization_1/beta/m/Read/ReadVariableOp8Adam/residual_unit/conv2d_2/kernel/m/Read/ReadVariableOpDAdam/residual_unit/batch_normalization_2/gamma/m/Read/ReadVariableOpCAdam/residual_unit/batch_normalization_2/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp8Adam/residual_unit/conv2d_1/kernel/v/Read/ReadVariableOpDAdam/residual_unit/batch_normalization_1/gamma/v/Read/ReadVariableOpCAdam/residual_unit/batch_normalization_1/beta/v/Read/ReadVariableOp8Adam/residual_unit/conv2d_2/kernel/v/Read/ReadVariableOpDAdam/residual_unit/batch_normalization_2/gamma/v/Read/ReadVariableOpCAdam/residual_unit/batch_normalization_2/beta/v/Read/ReadVariableOpConst*=
Tin6
422	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_287018
©
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasresidual_unit/conv2d_1/kernel)residual_unit/batch_normalization_1/gamma(residual_unit/batch_normalization_1/betaresidual_unit/conv2d_2/kernel)residual_unit/batch_normalization_2/gamma(residual_unit/batch_normalization_2/beta/residual_unit/batch_normalization_1/moving_mean3residual_unit/batch_normalization_1/moving_variance/residual_unit/batch_normalization_2/moving_mean3residual_unit/batch_normalization_2/moving_variance	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d/kernel/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense/kernel/mAdam/dense/bias/m$Adam/residual_unit/conv2d_1/kernel/m0Adam/residual_unit/batch_normalization_1/gamma/m/Adam/residual_unit/batch_normalization_1/beta/m$Adam/residual_unit/conv2d_2/kernel/m0Adam/residual_unit/batch_normalization_2/gamma/m/Adam/residual_unit/batch_normalization_2/beta/mAdam/conv2d/kernel/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense/kernel/vAdam/dense/bias/v$Adam/residual_unit/conv2d_1/kernel/v0Adam/residual_unit/batch_normalization_1/gamma/v/Adam/residual_unit/batch_normalization_1/beta/v$Adam/residual_unit/conv2d_2/kernel/v0Adam/residual_unit/batch_normalization_2/gamma/v/Adam/residual_unit/batch_normalization_2/beta/v*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_287172¦½
Ê

O__inference_batch_normalization_layer_call_and_return_conditional_losses_286515

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ð
¿
+__inference_sequential_layer_call_fn_285813
conv2d_input!
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
#
	unknown_9:



unknown_10:


unknown_11:


unknown_12:


unknown_13:


unknown_14:



unknown_15:

identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_285776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
(

F__inference_sequential_layer_call_and_return_conditional_losses_286022

inputs'
conv2d_285979:
(
batch_normalization_285982:
(
batch_normalization_285984:
(
batch_normalization_285986:
(
batch_normalization_285988:
.
residual_unit_285993:

"
residual_unit_285995:
"
residual_unit_285997:
"
residual_unit_285999:
"
residual_unit_286001:
.
residual_unit_286003:

"
residual_unit_286005:
"
residual_unit_286007:
"
residual_unit_286009:
"
residual_unit_286011:

dense_286016:


dense_286018:

identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢%residual_unit/StatefulPartitionedCallß
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_285979*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_285665ÿ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_285982batch_normalization_285984batch_normalization_285986batch_normalization_285988*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285487ò
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_285683ç
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_285507è
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_285993residual_unit_285995residual_unit_285997residual_unit_285999residual_unit_286001residual_unit_286003residual_unit_286005residual_unit_286007residual_unit_286009residual_unit_286011*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_unit_layer_call_and_return_conditional_losses_285898
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_285648á
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_285756þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_286016dense_286018*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_285769u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

'__inference_conv2d_layer_call_fn_286464

inputs!
unknown:

identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_285665w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
b
F__inference_activation_layer_call_and_return_conditional_losses_285683

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_285596

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_285507

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

³
B__inference_conv2d_layer_call_and_return_conditional_losses_285665

inputs8
conv2d_readvariableop_resource:

identity¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

O__inference_batch_normalization_layer_call_and_return_conditional_losses_285456

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¡(

F__inference_sequential_layer_call_and_return_conditional_losses_285776

inputs'
conv2d_285666:
(
batch_normalization_285669:
(
batch_normalization_285671:
(
batch_normalization_285673:
(
batch_normalization_285675:
.
residual_unit_285728:

"
residual_unit_285730:
"
residual_unit_285732:
"
residual_unit_285734:
"
residual_unit_285736:
.
residual_unit_285738:

"
residual_unit_285740:
"
residual_unit_285742:
"
residual_unit_285744:
"
residual_unit_285746:

dense_285770:


dense_285772:

identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢%residual_unit/StatefulPartitionedCallß
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_285666*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_285665
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_285669batch_normalization_285671batch_normalization_285673batch_normalization_285675*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285456ò
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_285683ç
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_285507ì
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_285728residual_unit_285730residual_unit_285732residual_unit_285734residual_unit_285736residual_unit_285738residual_unit_285740residual_unit_285742residual_unit_285744residual_unit_285746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_unit_layer_call_and_return_conditional_losses_285727
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_285648á
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_285756þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_285770dense_285772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_285769u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

U
9__inference_global_average_pooling2d_layer_call_fn_286690

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_285648i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_285648

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×]
Ó
F__inference_sequential_layer_call_and_return_conditional_losses_286386

inputs?
%conv2d_conv2d_readvariableop_resource:
9
+batch_normalization_readvariableop_resource:
;
-batch_normalization_readvariableop_1_resource:
J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:
L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:
O
5residual_unit_conv2d_1_conv2d_readvariableop_resource:

I
;residual_unit_batch_normalization_1_readvariableop_resource:
K
=residual_unit_batch_normalization_1_readvariableop_1_resource:
Z
Lresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
\
Nresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
O
5residual_unit_conv2d_2_conv2d_readvariableop_resource:

I
;residual_unit_batch_normalization_2_readvariableop_resource:
K
=residual_unit_batch_normalization_2_readvariableop_1_resource:
Z
Lresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
\
Nresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:
6
$dense_matmul_readvariableop_resource:

3
%dense_biasadd_readvariableop_resource:

identity¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢conv2d/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢2residual_unit/batch_normalization_1/ReadVariableOp¢4residual_unit/batch_normalization_1/ReadVariableOp_1¢Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢2residual_unit/batch_normalization_2/ReadVariableOp¢4residual_unit/batch_normalization_2/ReadVariableOp_1¢,residual_unit/conv2d_1/Conv2D/ReadVariableOp¢,residual_unit/conv2d_2/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:
*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:
*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0ª
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( {
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingSAME*
strides
ª
,residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0ß
residual_unit/conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:04residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
ª
2residual_unit/batch_normalization_1/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0®
4residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ì
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0Ð
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0
4residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_1/Conv2D:output:0:residual_unit/batch_normalization_1/ReadVariableOp:value:0<residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( 
residual_unit/ReluRelu8residual_unit/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
,residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0á
residual_unit/conv2d_2/Conv2DConv2D residual_unit/Relu:activations:04residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
ª
2residual_unit/batch_normalization_2/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0®
4residual_unit/batch_normalization_2/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ì
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0Ð
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0
4residual_unit/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_2/Conv2D:output:0:residual_unit/batch_normalization_2/ReadVariableOp:value:0<residual_unit/batch_normalization_2/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( ®
residual_unit/addAddV28residual_unit/batch_normalization_2/FusedBatchNormV3:y:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
residual_unit/Relu_1Reluresidual_unit/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      µ
global_average_pooling2d/MeanMean"residual_unit/Relu_1:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
®
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOpD^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_1/ReadVariableOp5^residual_unit/batch_normalization_1/ReadVariableOp_1D^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_2/ReadVariableOp5^residual_unit/batch_normalization_2/ReadVariableOp_1-^residual_unit/conv2d_1/Conv2D/ReadVariableOp-^residual_unit/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_1/ReadVariableOp2residual_unit/batch_normalization_1/ReadVariableOp2l
4residual_unit/batch_normalization_1/ReadVariableOp_14residual_unit/batch_normalization_1/ReadVariableOp_12
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_2/ReadVariableOp2residual_unit/batch_normalization_2/ReadVariableOp2l
4residual_unit/batch_normalization_2/ReadVariableOp_14residual_unit/batch_normalization_2/ReadVariableOp_12\
,residual_unit/conv2d_1/Conv2D/ReadVariableOp,residual_unit/conv2d_1/Conv2D/ReadVariableOp2\
,residual_unit/conv2d_2/Conv2D/ReadVariableOp,residual_unit/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
G
+__inference_activation_layer_call_fn_286538

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_285683h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
³
_
C__inference_flatten_layer_call_and_return_conditional_losses_286707

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285532

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ê
¿
+__inference_sequential_layer_call_fn_286098
conv2d_input!
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
#
	unknown_9:



unknown_10:


unknown_11:


unknown_12:


unknown_13:


unknown_14:



unknown_15:

identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_286022o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
	
Ï
4__inference_batch_normalization_layer_call_fn_286497

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285487
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¼

÷
.__inference_residual_unit_layer_call_fn_286603

inputs!
unknown:


	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_unit_layer_call_and_return_conditional_losses_285898w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Å
è!
"__inference__traced_restore_287172
file_prefix8
assignvariableop_conv2d_kernel:
:
,assignvariableop_1_batch_normalization_gamma:
9
+assignvariableop_2_batch_normalization_beta:
@
2assignvariableop_3_batch_normalization_moving_mean:
D
6assignvariableop_4_batch_normalization_moving_variance:
1
assignvariableop_5_dense_kernel:

+
assignvariableop_6_dense_bias:
J
0assignvariableop_7_residual_unit_conv2d_1_kernel:

J
<assignvariableop_8_residual_unit_batch_normalization_1_gamma:
I
;assignvariableop_9_residual_unit_batch_normalization_1_beta:
K
1assignvariableop_10_residual_unit_conv2d_2_kernel:

K
=assignvariableop_11_residual_unit_batch_normalization_2_gamma:
J
<assignvariableop_12_residual_unit_batch_normalization_2_beta:
Q
Cassignvariableop_13_residual_unit_batch_normalization_1_moving_mean:
U
Gassignvariableop_14_residual_unit_batch_normalization_1_moving_variance:
Q
Cassignvariableop_15_residual_unit_batch_normalization_2_moving_mean:
U
Gassignvariableop_16_residual_unit_batch_normalization_2_moving_variance:
'
assignvariableop_17_adam_iter:	 )
assignvariableop_18_adam_beta_1: )
assignvariableop_19_adam_beta_2: (
assignvariableop_20_adam_decay: 0
&assignvariableop_21_adam_learning_rate: %
assignvariableop_22_total_1: %
assignvariableop_23_count_1: #
assignvariableop_24_total: #
assignvariableop_25_count: B
(assignvariableop_26_adam_conv2d_kernel_m:
B
4assignvariableop_27_adam_batch_normalization_gamma_m:
A
3assignvariableop_28_adam_batch_normalization_beta_m:
9
'assignvariableop_29_adam_dense_kernel_m:

3
%assignvariableop_30_adam_dense_bias_m:
R
8assignvariableop_31_adam_residual_unit_conv2d_1_kernel_m:

R
Dassignvariableop_32_adam_residual_unit_batch_normalization_1_gamma_m:
Q
Cassignvariableop_33_adam_residual_unit_batch_normalization_1_beta_m:
R
8assignvariableop_34_adam_residual_unit_conv2d_2_kernel_m:

R
Dassignvariableop_35_adam_residual_unit_batch_normalization_2_gamma_m:
Q
Cassignvariableop_36_adam_residual_unit_batch_normalization_2_beta_m:
B
(assignvariableop_37_adam_conv2d_kernel_v:
B
4assignvariableop_38_adam_batch_normalization_gamma_v:
A
3assignvariableop_39_adam_batch_normalization_beta_v:
9
'assignvariableop_40_adam_dense_kernel_v:

3
%assignvariableop_41_adam_dense_bias_v:
R
8assignvariableop_42_adam_residual_unit_conv2d_1_kernel_v:

R
Dassignvariableop_43_adam_residual_unit_batch_normalization_1_gamma_v:
Q
Cassignvariableop_44_adam_residual_unit_batch_normalization_1_beta_v:
R
8assignvariableop_45_adam_residual_unit_conv2d_2_kernel_v:

R
Dassignvariableop_46_adam_residual_unit_batch_normalization_2_gamma_v:
Q
Cassignvariableop_47_adam_residual_unit_batch_normalization_2_beta_v:

identity_49¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9à
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueüBù1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp0assignvariableop_7_residual_unit_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_8AssignVariableOp<assignvariableop_8_residual_unit_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_9AssignVariableOp;assignvariableop_9_residual_unit_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_10AssignVariableOp1assignvariableop_10_residual_unit_conv2d_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_11AssignVariableOp=assignvariableop_11_residual_unit_batch_normalization_2_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_12AssignVariableOp<assignvariableop_12_residual_unit_batch_normalization_2_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_13AssignVariableOpCassignvariableop_13_residual_unit_batch_normalization_1_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_14AssignVariableOpGassignvariableop_14_residual_unit_batch_normalization_1_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_15AssignVariableOpCassignvariableop_15_residual_unit_batch_normalization_2_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_16AssignVariableOpGassignvariableop_16_residual_unit_batch_normalization_2_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_iterIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_batch_normalization_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_batch_normalization_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_31AssignVariableOp8assignvariableop_31_adam_residual_unit_conv2d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_32AssignVariableOpDassignvariableop_32_adam_residual_unit_batch_normalization_1_gamma_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_33AssignVariableOpCassignvariableop_33_adam_residual_unit_batch_normalization_1_beta_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_34AssignVariableOp8assignvariableop_34_adam_residual_unit_conv2d_2_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_35AssignVariableOpDassignvariableop_35_adam_residual_unit_batch_normalization_2_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_36AssignVariableOpCassignvariableop_36_adam_residual_unit_batch_normalization_2_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_conv2d_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_batch_normalization_gamma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_39AssignVariableOp3assignvariableop_39_adam_batch_normalization_beta_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_dense_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_42AssignVariableOp8assignvariableop_42_adam_residual_unit_conv2d_1_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_43AssignVariableOpDassignvariableop_43_adam_residual_unit_batch_normalization_1_gamma_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_44AssignVariableOpCassignvariableop_44_adam_residual_unit_batch_normalization_1_beta_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_45AssignVariableOp8assignvariableop_45_adam_residual_unit_conv2d_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_46AssignVariableOpDassignvariableop_46_adam_residual_unit_batch_normalization_2_gamma_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_47AssignVariableOpCassignvariableop_47_adam_residual_unit_batch_normalization_2_beta_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ï
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: Ü
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ä
¸
$__inference_signature_wrapper_286237
conv2d_input!
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
#
	unknown_9:



unknown_10:


unknown_11:


unknown_12:


unknown_13:


unknown_14:



unknown_15:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_285434o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
Çf

__inference__traced_save_287018
file_prefix,
(savev2_conv2d_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop<
8savev2_residual_unit_conv2d_1_kernel_read_readvariableopH
Dsavev2_residual_unit_batch_normalization_1_gamma_read_readvariableopG
Csavev2_residual_unit_batch_normalization_1_beta_read_readvariableop<
8savev2_residual_unit_conv2d_2_kernel_read_readvariableopH
Dsavev2_residual_unit_batch_normalization_2_gamma_read_readvariableopG
Csavev2_residual_unit_batch_normalization_2_beta_read_readvariableopN
Jsavev2_residual_unit_batch_normalization_1_moving_mean_read_readvariableopR
Nsavev2_residual_unit_batch_normalization_1_moving_variance_read_readvariableopN
Jsavev2_residual_unit_batch_normalization_2_moving_mean_read_readvariableopR
Nsavev2_residual_unit_batch_normalization_2_moving_variance_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableopC
?savev2_adam_residual_unit_conv2d_1_kernel_m_read_readvariableopO
Ksavev2_adam_residual_unit_batch_normalization_1_gamma_m_read_readvariableopN
Jsavev2_adam_residual_unit_batch_normalization_1_beta_m_read_readvariableopC
?savev2_adam_residual_unit_conv2d_2_kernel_m_read_readvariableopO
Ksavev2_adam_residual_unit_batch_normalization_2_gamma_m_read_readvariableopN
Jsavev2_adam_residual_unit_batch_normalization_2_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableopC
?savev2_adam_residual_unit_conv2d_1_kernel_v_read_readvariableopO
Ksavev2_adam_residual_unit_batch_normalization_1_gamma_v_read_readvariableopN
Jsavev2_adam_residual_unit_batch_normalization_1_beta_v_read_readvariableopC
?savev2_adam_residual_unit_conv2d_2_kernel_v_read_readvariableopO
Ksavev2_adam_residual_unit_batch_normalization_2_gamma_v_read_readvariableopN
Jsavev2_adam_residual_unit_batch_normalization_2_beta_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueüBù1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ó
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop8savev2_residual_unit_conv2d_1_kernel_read_readvariableopDsavev2_residual_unit_batch_normalization_1_gamma_read_readvariableopCsavev2_residual_unit_batch_normalization_1_beta_read_readvariableop8savev2_residual_unit_conv2d_2_kernel_read_readvariableopDsavev2_residual_unit_batch_normalization_2_gamma_read_readvariableopCsavev2_residual_unit_batch_normalization_2_beta_read_readvariableopJsavev2_residual_unit_batch_normalization_1_moving_mean_read_readvariableopNsavev2_residual_unit_batch_normalization_1_moving_variance_read_readvariableopJsavev2_residual_unit_batch_normalization_2_moving_mean_read_readvariableopNsavev2_residual_unit_batch_normalization_2_moving_variance_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop?savev2_adam_residual_unit_conv2d_1_kernel_m_read_readvariableopKsavev2_adam_residual_unit_batch_normalization_1_gamma_m_read_readvariableopJsavev2_adam_residual_unit_batch_normalization_1_beta_m_read_readvariableop?savev2_adam_residual_unit_conv2d_2_kernel_m_read_readvariableopKsavev2_adam_residual_unit_batch_normalization_2_gamma_m_read_readvariableopJsavev2_adam_residual_unit_batch_normalization_2_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop?savev2_adam_residual_unit_conv2d_1_kernel_v_read_readvariableopKsavev2_adam_residual_unit_batch_normalization_1_gamma_v_read_readvariableopJsavev2_adam_residual_unit_batch_normalization_1_beta_v_read_readvariableop?savev2_adam_residual_unit_conv2d_2_kernel_v_read_readvariableopKsavev2_adam_residual_unit_batch_normalization_2_gamma_v_read_readvariableopJsavev2_adam_residual_unit_batch_normalization_2_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesû
ø: :
:
:
:
:
:

:
:

:
:
:

:
:
:
:
:
:
: : : : : : : : : :
:
:
:

:
:

:
:
:

:
:
:
:
:
:

:
:

:
:
:

:
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 	

_output_shapes
:
: 


_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:, (
&
_output_shapes
:

: !

_output_shapes
:
: "

_output_shapes
:
:,#(
&
_output_shapes
:

: $

_output_shapes
:
: %

_output_shapes
:
:,&(
&
_output_shapes
:
: '

_output_shapes
:
: (

_output_shapes
:
:$) 

_output_shapes

:

: *

_output_shapes
:
:,+(
&
_output_shapes
:

: ,

_output_shapes
:
: -

_output_shapes
:
:,.(
&
_output_shapes
:

: /

_output_shapes
:
: 0

_output_shapes
:
:1

_output_shapes
: 
ºl
ª
!__inference__wrapped_model_285434
conv2d_inputJ
0sequential_conv2d_conv2d_readvariableop_resource:
D
6sequential_batch_normalization_readvariableop_resource:
F
8sequential_batch_normalization_readvariableop_1_resource:
U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:
W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:
Z
@sequential_residual_unit_conv2d_1_conv2d_readvariableop_resource:

T
Fsequential_residual_unit_batch_normalization_1_readvariableop_resource:
V
Hsequential_residual_unit_batch_normalization_1_readvariableop_1_resource:
e
Wsequential_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
g
Ysequential_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
Z
@sequential_residual_unit_conv2d_2_conv2d_readvariableop_resource:

T
Fsequential_residual_unit_batch_normalization_2_readvariableop_resource:
V
Hsequential_residual_unit_batch_normalization_2_readvariableop_1_resource:
e
Wsequential_residual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
g
Ysequential_residual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:
A
/sequential_dense_matmul_readvariableop_resource:

>
0sequential_dense_biasadd_readvariableop_resource:

identity¢>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp¢@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢-sequential/batch_normalization/ReadVariableOp¢/sequential/batch_normalization/ReadVariableOp_1¢'sequential/conv2d/Conv2D/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢Nsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢=sequential/residual_unit/batch_normalization_1/ReadVariableOp¢?sequential/residual_unit/batch_normalization_1/ReadVariableOp_1¢Nsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢=sequential/residual_unit/batch_normalization_2/ReadVariableOp¢?sequential/residual_unit/batch_normalization_2/ReadVariableOp_1¢7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp¢7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Ã
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
 
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:
*
dtype0¤
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:
*
dtype0Â
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0Æ
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0ì
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3!sequential/conv2d/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( 
sequential/activation/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/activation/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingSAME*
strides
À
7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@sequential_residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0
(sequential/residual_unit/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:0?sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
À
=sequential/residual_unit/batch_normalization_1/ReadVariableOpReadVariableOpFsequential_residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
?sequential/residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOpHsequential_residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0â
Nsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWsequential_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0æ
Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsequential_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ì
?sequential/residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV31sequential/residual_unit/conv2d_1/Conv2D:output:0Esequential/residual_unit/batch_normalization_1/ReadVariableOp:value:0Gsequential/residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Vsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Xsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( ¤
sequential/residual_unit/ReluReluCsequential/residual_unit/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@sequential_residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0
(sequential/residual_unit/conv2d_2/Conv2DConv2D+sequential/residual_unit/Relu:activations:0?sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
À
=sequential/residual_unit/batch_normalization_2/ReadVariableOpReadVariableOpFsequential_residual_unit_batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
?sequential/residual_unit/batch_normalization_2/ReadVariableOp_1ReadVariableOpHsequential_residual_unit_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0â
Nsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWsequential_residual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0æ
Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsequential_residual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ì
?sequential/residual_unit/batch_normalization_2/FusedBatchNormV3FusedBatchNormV31sequential/residual_unit/conv2d_2/Conv2D:output:0Esequential/residual_unit/batch_normalization_2/ReadVariableOp:value:0Gsequential/residual_unit/batch_normalization_2/ReadVariableOp_1:value:0Vsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Xsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( Ï
sequential/residual_unit/addAddV2Csequential/residual_unit/batch_normalization_2/FusedBatchNormV3:y:0)sequential/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential/residual_unit/Relu_1Relu sequential/residual_unit/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

:sequential/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ö
(sequential/global_average_pooling2d/MeanMean-sequential/residual_unit/Relu_1:activations:0Csequential/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   ­
sequential/flatten/ReshapeReshape1sequential/global_average_pooling2d/Mean:output:0!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0¨
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
IdentityIdentity"sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
é
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOpO^sequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpQ^sequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>^sequential/residual_unit/batch_normalization_1/ReadVariableOp@^sequential/residual_unit/batch_normalization_1/ReadVariableOp_1O^sequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpQ^sequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1>^sequential/residual_unit/batch_normalization_2/ReadVariableOp@^sequential/residual_unit/batch_normalization_2/ReadVariableOp_18^sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp8^sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2 
Nsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpNsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2¤
Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12~
=sequential/residual_unit/batch_normalization_1/ReadVariableOp=sequential/residual_unit/batch_normalization_1/ReadVariableOp2
?sequential/residual_unit/batch_normalization_1/ReadVariableOp_1?sequential/residual_unit/batch_normalization_1/ReadVariableOp_12 
Nsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpNsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2¤
Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12~
=sequential/residual_unit/batch_normalization_2/ReadVariableOp=sequential/residual_unit/batch_normalization_2/ReadVariableOp2
?sequential/residual_unit/batch_normalization_2/ReadVariableOp_1?sequential/residual_unit/batch_normalization_2/ReadVariableOp_12r
7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp2r
7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
³
_
C__inference_flatten_layer_call_and_return_conditional_losses_285756

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

¾
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285487

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

À
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286851

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ì>

I__inference_residual_unit_layer_call_and_return_conditional_losses_285898

inputsA
'conv2d_1_conv2d_readvariableop_resource:

;
-batch_normalization_1_readvariableop_resource:
=
/batch_normalization_1_readvariableop_1_resource:
L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
A
'conv2d_2_conv2d_readvariableop_resource:

;
-batch_normalization_2_readvariableop_resource:
=
/batch_normalization_2_readvariableop_1_resource:
L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:

identity¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0«
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ä
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0·
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ä
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


ò
A__inference_dense_layer_call_and_return_conditional_losses_286727

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
À

÷
.__inference_residual_unit_layer_call_fn_286578

inputs!
unknown:


	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:

identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_unit_layer_call_and_return_conditional_losses_285727w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
´
J
.__inference_max_pooling2d_layer_call_fn_286548

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_285507
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_2_layer_call_fn_286802

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_285596
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¼

&__inference_dense_layer_call_fn_286716

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_285769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_2_layer_call_fn_286815

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_285627
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
µ
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_286696

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_1_layer_call_fn_286740

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285532
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_286553

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286771

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286833

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

D
(__inference_flatten_layer_call_fn_286701

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_285756`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

À
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285563

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¸0
è	
I__inference_residual_unit_layer_call_and_return_conditional_losses_286644

inputsA
'conv2d_1_conv2d_readvariableop_resource:

;
-batch_normalization_1_readvariableop_resource:
=
/batch_normalization_1_readvariableop_1_resource:
L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
A
'conv2d_2_conv2d_readvariableop_resource:

;
-batch_normalization_2_readvariableop_resource:
=
/batch_normalization_2_readvariableop_1_resource:
L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:

identity¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0«
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0¶
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0·
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0¶
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ø
¹
+__inference_sequential_layer_call_fn_286315

inputs!
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
#
	unknown_9:



unknown_10:


unknown_11:


unknown_12:


unknown_13:


unknown_14:



unknown_15:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_286022o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

³
B__inference_conv2d_layer_call_and_return_conditional_losses_286471

inputs8
conv2d_readvariableop_resource:

identity¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_1_layer_call_fn_286753

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_285563
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ì>

I__inference_residual_unit_layer_call_and_return_conditional_losses_286685

inputsA
'conv2d_1_conv2d_readvariableop_resource:

;
-batch_normalization_1_readvariableop_resource:
=
/batch_normalization_1_readvariableop_1_resource:
L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
A
'conv2d_2_conv2d_readvariableop_resource:

;
-batch_normalization_2_readvariableop_resource:
=
/batch_normalization_2_readvariableop_1_resource:
L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:

identity¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0«
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ä
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0·
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ä
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Þ
¹
+__inference_sequential_layer_call_fn_286276

inputs!
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:
#
	unknown_9:



unknown_10:


unknown_11:


unknown_12:


unknown_13:


unknown_14:



unknown_15:

identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_285776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³(

F__inference_sequential_layer_call_and_return_conditional_losses_286144
conv2d_input'
conv2d_286101:
(
batch_normalization_286104:
(
batch_normalization_286106:
(
batch_normalization_286108:
(
batch_normalization_286110:
.
residual_unit_286115:

"
residual_unit_286117:
"
residual_unit_286119:
"
residual_unit_286121:
"
residual_unit_286123:
.
residual_unit_286125:

"
residual_unit_286127:
"
residual_unit_286129:
"
residual_unit_286131:
"
residual_unit_286133:

dense_286138:


dense_286140:

identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢%residual_unit/StatefulPartitionedCallå
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_286101*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_285665
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_286104batch_normalization_286106batch_normalization_286108batch_normalization_286110*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285456ò
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_285683ç
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_285507ì
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_286115residual_unit_286117residual_unit_286119residual_unit_286121residual_unit_286123residual_unit_286125residual_unit_286127residual_unit_286129residual_unit_286131residual_unit_286133*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_unit_layer_call_and_return_conditional_losses_285727
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_285648á
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_285756þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_286138dense_286140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_285769u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input


ò
A__inference_dense_layer_call_and_return_conditional_losses_285769

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

¾
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286533

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ê
b
F__inference_activation_layer_call_and_return_conditional_losses_286543

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

À
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286789

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Ï
4__inference_batch_normalization_layer_call_fn_286484

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285456
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Åv
÷
F__inference_sequential_layer_call_and_return_conditional_losses_286457

inputs?
%conv2d_conv2d_readvariableop_resource:
9
+batch_normalization_readvariableop_resource:
;
-batch_normalization_readvariableop_1_resource:
J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:
L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:
O
5residual_unit_conv2d_1_conv2d_readvariableop_resource:

I
;residual_unit_batch_normalization_1_readvariableop_resource:
K
=residual_unit_batch_normalization_1_readvariableop_1_resource:
Z
Lresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
\
Nresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
O
5residual_unit_conv2d_2_conv2d_readvariableop_resource:

I
;residual_unit_batch_normalization_2_readvariableop_resource:
K
=residual_unit_batch_normalization_2_readvariableop_1_resource:
Z
Lresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
\
Nresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:
6
$dense_matmul_readvariableop_resource:

3
%dense_biasadd_readvariableop_resource:

identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢conv2d/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢2residual_unit/batch_normalization_1/AssignNewValue¢4residual_unit/batch_normalization_1/AssignNewValue_1¢Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢2residual_unit/batch_normalization_1/ReadVariableOp¢4residual_unit/batch_normalization_1/ReadVariableOp_1¢2residual_unit/batch_normalization_2/AssignNewValue¢4residual_unit/batch_normalization_2/AssignNewValue_1¢Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢2residual_unit/batch_normalization_2/ReadVariableOp¢4residual_unit/batch_normalization_2/ReadVariableOp_1¢,residual_unit/conv2d_1/Conv2D/ReadVariableOp¢,residual_unit/conv2d_2/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:
*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:
*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0¸
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape( 
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingSAME*
strides
ª
,residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0ß
residual_unit/conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:04residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
ª
2residual_unit/batch_normalization_1/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0®
4residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ì
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0Ð
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0
4residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_1/Conv2D:output:0:residual_unit/batch_normalization_1/ReadVariableOp:value:0<residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Ö
2residual_unit/batch_normalization_1/AssignNewValueAssignVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceAresidual_unit/batch_normalization_1/FusedBatchNormV3:batch_mean:0D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(à
4residual_unit/batch_normalization_1/AssignNewValue_1AssignVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceEresidual_unit/batch_normalization_1/FusedBatchNormV3:batch_variance:0F^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
residual_unit/ReluRelu8residual_unit/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
,residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0á
residual_unit/conv2d_2/Conv2DConv2D residual_unit/Relu:activations:04residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides
ª
2residual_unit/batch_normalization_2/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0®
4residual_unit/batch_normalization_2/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ì
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0Ð
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0
4residual_unit/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_2/Conv2D:output:0:residual_unit/batch_normalization_2/ReadVariableOp:value:0<residual_unit/batch_normalization_2/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Ö
2residual_unit/batch_normalization_2/AssignNewValueAssignVariableOpLresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceAresidual_unit/batch_normalization_2/FusedBatchNormV3:batch_mean:0D^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(à
4residual_unit/batch_normalization_2/AssignNewValue_1AssignVariableOpNresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceEresidual_unit/batch_normalization_2/FusedBatchNormV3:batch_variance:0F^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(®
residual_unit/addAddV28residual_unit/batch_normalization_2/FusedBatchNormV3:y:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
residual_unit/Relu_1Reluresidual_unit/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      µ
global_average_pooling2d/MeanMean"residual_unit/Relu_1:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ
   
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ò	
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp3^residual_unit/batch_normalization_1/AssignNewValue5^residual_unit/batch_normalization_1/AssignNewValue_1D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_1/ReadVariableOp5^residual_unit/batch_normalization_1/ReadVariableOp_13^residual_unit/batch_normalization_2/AssignNewValue5^residual_unit/batch_normalization_2/AssignNewValue_1D^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_2/ReadVariableOp5^residual_unit/batch_normalization_2/ReadVariableOp_1-^residual_unit/conv2d_1/Conv2D/ReadVariableOp-^residual_unit/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2h
2residual_unit/batch_normalization_1/AssignNewValue2residual_unit/batch_normalization_1/AssignNewValue2l
4residual_unit/batch_normalization_1/AssignNewValue_14residual_unit/batch_normalization_1/AssignNewValue_12
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_1/ReadVariableOp2residual_unit/batch_normalization_1/ReadVariableOp2l
4residual_unit/batch_normalization_1/ReadVariableOp_14residual_unit/batch_normalization_1/ReadVariableOp_12h
2residual_unit/batch_normalization_2/AssignNewValue2residual_unit/batch_normalization_2/AssignNewValue2l
4residual_unit/batch_normalization_2/AssignNewValue_14residual_unit/batch_normalization_2/AssignNewValue_12
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_2/ReadVariableOp2residual_unit/batch_normalization_2/ReadVariableOp2l
4residual_unit/batch_normalization_2/ReadVariableOp_14residual_unit/batch_normalization_2/ReadVariableOp_12\
,residual_unit/conv2d_1/Conv2D/ReadVariableOp,residual_unit/conv2d_1/Conv2D/ReadVariableOp2\
,residual_unit/conv2d_2/Conv2D/ReadVariableOp,residual_unit/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­(

F__inference_sequential_layer_call_and_return_conditional_losses_286190
conv2d_input'
conv2d_286147:
(
batch_normalization_286150:
(
batch_normalization_286152:
(
batch_normalization_286154:
(
batch_normalization_286156:
.
residual_unit_286161:

"
residual_unit_286163:
"
residual_unit_286165:
"
residual_unit_286167:
"
residual_unit_286169:
.
residual_unit_286171:

"
residual_unit_286173:
"
residual_unit_286175:
"
residual_unit_286177:
"
residual_unit_286179:

dense_286184:


dense_286186:

identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢%residual_unit/StatefulPartitionedCallå
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_286147*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_285665ÿ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_286150batch_normalization_286152batch_normalization_286154batch_normalization_286156*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_285487ò
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_285683ç
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_285507è
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_286161residual_unit_286163residual_unit_286165residual_unit_286167residual_unit_286169residual_unit_286171residual_unit_286173residual_unit_286175residual_unit_286177residual_unit_286179*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_unit_layer_call_and_return_conditional_losses_285898
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_285648á
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_285756þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_286184dense_286186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_285769u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input

À
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_285627

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¸0
è	
I__inference_residual_unit_layer_call_and_return_conditional_losses_285727

inputsA
'conv2d_1_conv2d_readvariableop_resource:

;
-batch_normalization_1_readvariableop_resource:
=
/batch_normalization_1_readvariableop_1_resource:
L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
A
'conv2d_2_conv2d_readvariableop_resource:

;
-batch_normalization_2_readvariableop_resource:
=
/batch_normalization_2_readvariableop_1_resource:
L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:
N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:

identity¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0«
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0¶
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0·
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingSAME*
strides

$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0°
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0´
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0¶
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
:
:
:
:
:*
epsilon%o:*
is_training( z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
M
conv2d_input=
serving_default_conv2d_input:0ÿÿÿÿÿÿÿÿÿ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:Ï·
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ó
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 _jit_compiled_convolution_op"
_tf_keras_layer
ê
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 axis
	!gamma
"beta
#moving_mean
$moving_variance"
_tf_keras_layer
¥
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
Ç
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7main_layers
8skip_layers"
_tf_keras_layer
¥
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
»
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias"
_tf_keras_layer

0
!1
"2
#3
$4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
K15
L16"
trackable_list_wrapper
n
0
!1
"2
M3
N4
O5
P6
Q7
R8
K9
L10"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
á
\trace_0
]trace_1
^trace_2
_trace_32ö
+__inference_sequential_layer_call_fn_285813
+__inference_sequential_layer_call_fn_286276
+__inference_sequential_layer_call_fn_286315
+__inference_sequential_layer_call_fn_286098¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z\trace_0z]trace_1z^trace_2z_trace_3
Í
`trace_0
atrace_1
btrace_2
ctrace_32â
F__inference_sequential_layer_call_and_return_conditional_losses_286386
F__inference_sequential_layer_call_and_return_conditional_losses_286457
F__inference_sequential_layer_call_and_return_conditional_losses_286144
F__inference_sequential_layer_call_and_return_conditional_losses_286190¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z`trace_0zatrace_1zbtrace_2zctrace_3
ÑBÎ
!__inference__wrapped_model_285434conv2d_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¯
diter

ebeta_1

fbeta_2
	gdecay
hlearning_ratemí!mî"mïKmðLmñMmòNmóOmôPmõQmöRm÷vø!vù"vúKvûLvüMvýNvþOvÿPvQvRv"
	optimizer
,
iserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
otrace_02Î
'__inference_conv2d_layer_call_fn_286464¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0

ptrace_02é
B__inference_conv2d_layer_call_and_return_conditional_losses_286471¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zptrace_0
':%
2conv2d/kernel
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
!0
"1
#2
$3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ù
vtrace_0
wtrace_12¢
4__inference_batch_normalization_layer_call_fn_286484
4__inference_batch_normalization_layer_call_fn_286497³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0zwtrace_1

xtrace_0
ytrace_12Ø
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286515
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286533³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zxtrace_0zytrace_1
 "
trackable_list_wrapper
':%
2batch_normalization/gamma
&:$
2batch_normalization/beta
/:-
 (2batch_normalization/moving_mean
3:1
 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ò
+__inference_activation_layer_call_fn_286538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_activation_layer_call_and_return_conditional_losses_286543¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
ô
trace_02Õ
.__inference_max_pooling2d_layer_call_fn_286548¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ð
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_286553¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
f
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9"
trackable_list_wrapper
J
M0
N1
O2
P3
Q4
R5"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Ý
trace_0
trace_12¢
.__inference_residual_unit_layer_call_fn_286578
.__inference_residual_unit_layer_call_fn_286603¿
¶²²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ø
I__inference_residual_unit_layer_call_and_return_conditional_losses_286644
I__inference_residual_unit_layer_call_and_return_conditional_losses_286685¿
¶²²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ztrace_0ztrace_1
@
0
1
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
ÿ
trace_02à
9__inference_global_average_pooling2d_layer_call_fn_286690¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02û
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_286696¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
î
¡trace_02Ï
(__inference_flatten_layer_call_fn_286701¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¡trace_0

¢trace_02ê
C__inference_flatten_layer_call_and_return_conditional_losses_286707¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¢trace_0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ì
¨trace_02Í
&__inference_dense_layer_call_fn_286716¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¨trace_0

©trace_02è
A__inference_dense_layer_call_and_return_conditional_losses_286727¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z©trace_0
:

2dense/kernel
:
2
dense/bias
7:5

2residual_unit/conv2d_1/kernel
7:5
2)residual_unit/batch_normalization_1/gamma
6:4
2(residual_unit/batch_normalization_1/beta
7:5

2residual_unit/conv2d_2/kernel
7:5
2)residual_unit/batch_normalization_2/gamma
6:4
2(residual_unit/batch_normalization_2/beta
?:=
 (2/residual_unit/batch_normalization_1/moving_mean
C:A
 (23residual_unit/batch_normalization_1/moving_variance
?:=
 (2/residual_unit/batch_normalization_2/moving_mean
C:A
 (23residual_unit/batch_normalization_2/moving_variance
J
#0
$1
S2
T3
U4
V5"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
ª0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bÿ
+__inference_sequential_layer_call_fn_285813conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
+__inference_sequential_layer_call_fn_286276inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
+__inference_sequential_layer_call_fn_286315inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
+__inference_sequential_layer_call_fn_286098conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_286386inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_286457inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_286144conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_286190conv2d_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÐBÍ
$__inference_signature_wrapper_286237conv2d_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_conv2d_layer_call_fn_286464inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_conv2d_layer_call_and_return_conditional_losses_286471inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ùBö
4__inference_batch_normalization_layer_call_fn_286484inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
4__inference_batch_normalization_layer_call_fn_286497inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286515inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286533inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_activation_layer_call_fn_286538inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_activation_layer_call_and_return_conditional_losses_286543inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_max_pooling2d_layer_call_fn_286548inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_286553inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<
S0
T1
U2
V3"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÿBü
.__inference_residual_unit_layer_call_fn_286578inputs"¿
¶²²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
ÿBü
.__inference_residual_unit_layer_call_fn_286603inputs"¿
¶²²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_residual_unit_layer_call_and_return_conditional_losses_286644inputs"¿
¶²²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
I__inference_residual_unit_layer_call_and_return_conditional_losses_286685inputs"¿
¶²²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ú
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°__call__
+±&call_and_return_all_conditional_losses

Mkernel
!²_jit_compiled_convolution_op"
_tf_keras_layer
ñ
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses
	¹axis
	Ngamma
Obeta
Smoving_mean
Tmoving_variance"
_tf_keras_layer
Ú
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses

Pkernel
!À_jit_compiled_convolution_op"
_tf_keras_layer
ñ
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses
	Çaxis
	Qgamma
Rbeta
Umoving_mean
Vmoving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
íBê
9__inference_global_average_pooling2d_layer_call_fn_286690inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_286696inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_flatten_layer_call_fn_286701inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_flatten_layer_call_and_return_conditional_losses_286707inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_dense_layer_call_fn_286716inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_dense_layer_call_and_return_conditional_losses_286727inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
È	variables
É	keras_api

Êtotal

Ëcount"
_tf_keras_metric
c
Ì	variables
Í	keras_api

Îtotal

Ïcount
Ð
_fn_kwargs"
_tf_keras_metric
'
M0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
¬	variables
­trainable_variables
®regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
N0
O1
S2
T3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
á
Ûtrace_0
Ütrace_12¦
6__inference_batch_normalization_1_layer_call_fn_286740
6__inference_batch_normalization_1_layer_call_fn_286753³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÛtrace_0zÜtrace_1

Ýtrace_0
Þtrace_12Ü
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286771
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286789³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÝtrace_0zÞtrace_1
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
Q0
R1
U2
V3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
á
étrace_0
êtrace_12¦
6__inference_batch_normalization_2_layer_call_fn_286802
6__inference_batch_normalization_2_layer_call_fn_286815³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zétrace_0zêtrace_1

ëtrace_0
ìtrace_12Ü
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286833
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286851³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zëtrace_0zìtrace_1
 "
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
.
È	variables"
_generic_user_object
:  (2total
:  (2count
0
Î0
Ï1"
trackable_list_wrapper
.
Ì	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
6__inference_batch_normalization_1_layer_call_fn_286740inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_1_layer_call_fn_286753inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286771inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286789inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
6__inference_batch_normalization_2_layer_call_fn_286802inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
6__inference_batch_normalization_2_layer_call_fn_286815inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286833inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286851inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*
2Adam/conv2d/kernel/m
,:*
2 Adam/batch_normalization/gamma/m
+:)
2Adam/batch_normalization/beta/m
#:!

2Adam/dense/kernel/m
:
2Adam/dense/bias/m
<::

2$Adam/residual_unit/conv2d_1/kernel/m
<::
20Adam/residual_unit/batch_normalization_1/gamma/m
;:9
2/Adam/residual_unit/batch_normalization_1/beta/m
<::

2$Adam/residual_unit/conv2d_2/kernel/m
<::
20Adam/residual_unit/batch_normalization_2/gamma/m
;:9
2/Adam/residual_unit/batch_normalization_2/beta/m
,:*
2Adam/conv2d/kernel/v
,:*
2 Adam/batch_normalization/gamma/v
+:)
2Adam/batch_normalization/beta/v
#:!

2Adam/dense/kernel/v
:
2Adam/dense/bias/v
<::

2$Adam/residual_unit/conv2d_1/kernel/v
<::
20Adam/residual_unit/batch_normalization_1/gamma/v
;:9
2/Adam/residual_unit/batch_normalization_1/beta/v
<::

2$Adam/residual_unit/conv2d_2/kernel/v
<::
20Adam/residual_unit/batch_normalization_2/gamma/v
;:9
2/Adam/residual_unit/batch_normalization_2/beta/v§
!__inference__wrapped_model_285434!"#$MNOSTPQRUVKL=¢:
3¢0
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ
²
F__inference_activation_layer_call_and_return_conditional_losses_286543h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
+__inference_activation_layer_call_fn_286538[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª " ÿÿÿÿÿÿÿÿÿ
ì
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286771NOSTM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 ì
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_286789NOSTM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 Ä
6__inference_batch_normalization_1_layer_call_fn_286740NOSTM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ä
6__inference_batch_normalization_1_layer_call_fn_286753NOSTM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286833QRUVM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 ì
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_286851QRUVM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 Ä
6__inference_batch_normalization_2_layer_call_fn_286802QRUVM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ä
6__inference_batch_normalization_2_layer_call_fn_286815QRUVM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286515!"#$M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 ê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_286533!"#$M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 Â
4__inference_batch_normalization_layer_call_fn_286484!"#$M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Â
4__inference_batch_normalization_layer_call_fn_286497!"#$M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
±
B__inference_conv2d_layer_call_and_return_conditional_losses_286471k7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
'__inference_conv2d_layer_call_fn_286464^7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ
¡
A__inference_dense_layer_call_and_return_conditional_losses_286727\KL/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 y
&__inference_dense_layer_call_fn_286716OKL/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ

C__inference_flatten_layer_call_and_return_conditional_losses_286707X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 w
(__inference_flatten_layer_call_fn_286701K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
Ý
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_286696R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ´
9__inference_global_average_pooling2d_layer_call_fn_286690wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_286553R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_286548R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
I__inference_residual_unit_layer_call_and_return_conditional_losses_286644
MNOSTPQRUVG¢D
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 Ò
I__inference_residual_unit_layer_call_and_return_conditional_losses_286685
MNOSTPQRUVG¢D
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp"-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 ©
.__inference_residual_unit_layer_call_fn_286578w
MNOSTPQRUVG¢D
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp " ÿÿÿÿÿÿÿÿÿ
©
.__inference_residual_unit_layer_call_fn_286603w
MNOSTPQRUVG¢D
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª

trainingp" ÿÿÿÿÿÿÿÿÿ
Ì
F__inference_sequential_layer_call_and_return_conditional_losses_286144!"#$MNOSTPQRUVKLE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ì
F__inference_sequential_layer_call_and_return_conditional_losses_286190!"#$MNOSTPQRUVKLE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Å
F__inference_sequential_layer_call_and_return_conditional_losses_286386{!"#$MNOSTPQRUVKL?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Å
F__inference_sequential_layer_call_and_return_conditional_losses_286457{!"#$MNOSTPQRUVKL?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 £
+__inference_sequential_layer_call_fn_285813t!"#$MNOSTPQRUVKLE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
£
+__inference_sequential_layer_call_fn_286098t!"#$MNOSTPQRUVKLE¢B
;¢8
.+
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

+__inference_sequential_layer_call_fn_286276n!"#$MNOSTPQRUVKL?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

+__inference_sequential_layer_call_fn_286315n!"#$MNOSTPQRUVKL?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
º
$__inference_signature_wrapper_286237!"#$MNOSTPQRUVKLM¢J
¢ 
Cª@
>
conv2d_input.+
conv2d_inputÿÿÿÿÿÿÿÿÿ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ
