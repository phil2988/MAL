Р╪
Г╙
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ы
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
√
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
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
В
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
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ъЧ
╛
3residual_unit/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53residual_unit/batch_normalization_2/moving_variance
╖
Gresidual_unit/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp3residual_unit/batch_normalization_2/moving_variance*
_output_shapes
:
*
dtype0
╢
/residual_unit/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/residual_unit/batch_normalization_2/moving_mean
п
Cresidual_unit/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp/residual_unit/batch_normalization_2/moving_mean*
_output_shapes
:
*
dtype0
╛
3residual_unit/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53residual_unit/batch_normalization_1/moving_variance
╖
Gresidual_unit/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp3residual_unit/batch_normalization_1/moving_variance*
_output_shapes
:
*
dtype0
╢
/residual_unit/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/residual_unit/batch_normalization_1/moving_mean
п
Cresidual_unit/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp/residual_unit/batch_normalization_1/moving_mean*
_output_shapes
:
*
dtype0
и
(residual_unit/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(residual_unit/batch_normalization_2/beta
б
<residual_unit/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp(residual_unit/batch_normalization_2/beta*
_output_shapes
:
*
dtype0
к
)residual_unit/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)residual_unit/batch_normalization_2/gamma
г
=residual_unit/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp)residual_unit/batch_normalization_2/gamma*
_output_shapes
:
*
dtype0
Ю
residual_unit/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*.
shared_nameresidual_unit/conv2d_2/kernel
Ч
1residual_unit/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpresidual_unit/conv2d_2/kernel*&
_output_shapes
:

*
dtype0
и
(residual_unit/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(residual_unit/batch_normalization_1/beta
б
<residual_unit/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp(residual_unit/batch_normalization_1/beta*
_output_shapes
:
*
dtype0
к
)residual_unit/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)residual_unit/batch_normalization_1/gamma
г
=residual_unit/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp)residual_unit/batch_normalization_1/gamma*
_output_shapes
:
*
dtype0
Ю
residual_unit/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*.
shared_nameresidual_unit/conv2d_1/kernel
Ч
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
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:
*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:
*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:
*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namebatch_normalization/gamma
Г
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
П
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
п
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceresidual_unit/conv2d_1/kernel)residual_unit/batch_normalization_1/gamma(residual_unit/batch_normalization_1/beta/residual_unit/batch_normalization_1/moving_mean3residual_unit/batch_normalization_1/moving_varianceresidual_unit/conv2d_2/kernel)residual_unit/batch_normalization_2/gamma(residual_unit/batch_normalization_2/beta/residual_unit/batch_normalization_2/moving_mean3residual_unit/batch_normalization_2/moving_variancedense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_1079

NoOpNoOp
┘O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ФO
valueКOBЗO BАO
Н
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

signatures*
╛
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 _jit_compiled_convolution_op*
╒
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	 gamma
!beta
"moving_mean
#moving_variance*
О
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
О
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
▓
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6main_layers
7skip_layers*
О
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
О
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
ж
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias*
В
0
 1
!2
"3
#4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
J15
K16*
R
0
 1
!2
L3
M4
N5
O6
P7
Q8
J9
K10*
* 
░
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
[trace_0
\trace_1
]trace_2
^trace_3* 
6
_trace_0
`trace_1
atrace_2
btrace_3* 
* 

cserving_default* 

0*

0*
* 
У
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
 0
!1
"2
#3*

 0
!1*
* 
У
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ptrace_0
qtrace_1* 

rtrace_0
strace_1* 
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
С
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

ytrace_0* 

ztrace_0* 
* 
* 
* 
С
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

Аtrace_0* 

Бtrace_0* 
J
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9*
.
L0
M1
N2
O3
P4
Q5*
* 
Ш
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Зtrace_0
Иtrace_1* 

Йtrace_0
Кtrace_1* 
$
Л0
М1
Н3
О4*
* 
* 
* 
* 
Ц
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Фtrace_0* 

Хtrace_0* 
* 
* 
* 
Ц
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

Ыtrace_0* 

Ьtrace_0* 

J0
K1*

J0
K1*
* 
Ш
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

вtrace_0* 

гtrace_0* 
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
"0
#1
R2
S3
T4
U5*
<
0
1
2
3
4
5
6
7*
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

"0
#1*
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
 
R0
S1
T2
U3*
$
Л0
М1
Н2
О3*
* 
* 
* 
* 
* 
* 
* 
┼
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses

Lkernel
!к_jit_compiled_convolution_op*
▄
л	variables
мtrainable_variables
нregularization_losses
о	keras_api
п__call__
+░&call_and_return_all_conditional_losses
	▒axis
	Mgamma
Nbeta
Rmoving_mean
Smoving_variance*
┼
▓	variables
│trainable_variables
┤regularization_losses
╡	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses

Okernel
!╕_jit_compiled_convolution_op*
▄
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses
	┐axis
	Pgamma
Qbeta
Tmoving_mean
Umoving_variance*
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

L0*

L0*
* 
Ю
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses*
* 
* 
* 
 
M0
N1
R2
S3*

M0
N1*
* 
Ю
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
л	variables
мtrainable_variables
нregularization_losses
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses*

╩trace_0
╦trace_1* 

╠trace_0
═trace_1* 
* 

O0*

O0*
* 
Ю
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses*
* 
* 
* 
 
P0
Q1
T2
U3*

P0
Q1*
* 
Ю
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
╣	variables
║trainable_variables
╗regularization_losses
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses*

╪trace_0
┘trace_1* 

┌trace_0
█trace_1* 
* 
* 
* 
* 
* 
* 

R0
S1*
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
T0
U1*
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╓	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp1residual_unit/conv2d_1/kernel/Read/ReadVariableOp=residual_unit/batch_normalization_1/gamma/Read/ReadVariableOp<residual_unit/batch_normalization_1/beta/Read/ReadVariableOp1residual_unit/conv2d_2/kernel/Read/ReadVariableOp=residual_unit/batch_normalization_2/gamma/Read/ReadVariableOp<residual_unit/batch_normalization_2/beta/Read/ReadVariableOpCresidual_unit/batch_normalization_1/moving_mean/Read/ReadVariableOpGresidual_unit/batch_normalization_1/moving_variance/Read/ReadVariableOpCresidual_unit/batch_normalization_2/moving_mean/Read/ReadVariableOpGresidual_unit/batch_normalization_2/moving_variance/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8В *&
f!R
__inference__traced_save_1767
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasresidual_unit/conv2d_1/kernel)residual_unit/batch_normalization_1/gamma(residual_unit/batch_normalization_1/betaresidual_unit/conv2d_2/kernel)residual_unit/batch_normalization_2/gamma(residual_unit/batch_normalization_2/beta/residual_unit/batch_normalization_1/moving_mean3residual_unit/batch_normalization_1/moving_variance/residual_unit/batch_normalization_2/moving_mean3residual_unit/batch_normalization_2/moving_variance*
Tin
2*
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
GPU 2J 8В *)
f$R"
 __inference__traced_restore_1828╞Д

ч
_
C__inference_activation_layer_call_and_return_conditional_losses_531

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         
b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
Г
╜
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_411

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Ж	
═
2__inference_batch_normalization_layer_call_fn_1339

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_335Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
╢0
ц	
G__inference_residual_unit_layer_call_and_return_conditional_losses_1486

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
identityИв5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1вconv2d_1/Conv2D/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpО
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0л
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╢
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╖
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╢
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:         
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:         
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:         
М
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         
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
:         

 
_user_specified_nameinputs
╗

ї
,__inference_residual_unit_layer_call_fn_1420

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
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_residual_unit_layer_call_and_return_conditional_losses_575w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
╡0
х	
F__inference_residual_unit_layer_call_and_return_conditional_losses_575

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
identityИв5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1вconv2d_1/Conv2D/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpО
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0л
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╢
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╖
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╢
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:         
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:         
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:         
М
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         
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
:         

 
_user_specified_nameinputs
я.
╤	
__inference__traced_save_1767
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
Nsavev2_residual_unit_batch_normalization_2_moving_variance_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ░
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┘
value╧B╠B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHС
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B ф	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop8savev2_residual_unit_conv2d_1_kernel_read_readvariableopDsavev2_residual_unit_batch_normalization_1_gamma_read_readvariableopCsavev2_residual_unit_batch_normalization_1_beta_read_readvariableop8savev2_residual_unit_conv2d_2_kernel_read_readvariableopDsavev2_residual_unit_batch_normalization_2_gamma_read_readvariableopCsavev2_residual_unit_batch_normalization_2_beta_read_readvariableopJsavev2_residual_unit_batch_normalization_1_moving_mean_read_readvariableopNsavev2_residual_unit_batch_normalization_1_moving_variance_read_readvariableopJsavev2_residual_unit_batch_normalization_2_moving_mean_read_readvariableopNsavev2_residual_unit_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*з
_input_shapesХ
Т: :
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
: 
И	
═
2__inference_batch_normalization_layer_call_fn_1326

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_304Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Д
╛
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1693

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
М	
╧
4__inference_batch_normalization_1_layer_call_fn_1582

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_380Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
п
H
,__inference_max_pooling2d_layer_call_fn_1390

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_355Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╙
╖
)__inference_sequential_layer_call_fn_1157

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
identityИвStatefulPartitionedCallЫ
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
:         
*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Г
╜
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_475

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
В
╝
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1375

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
╩
Ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1613

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Ъ'
╙
C__inference_sequential_layer_call_and_return_conditional_losses_870

inputs$

conv2d_827:
%
batch_normalization_830:
%
batch_normalization_832:
%
batch_normalization_834:
%
batch_normalization_836:
+
residual_unit_841:


residual_unit_843:

residual_unit_845:

residual_unit_847:

residual_unit_849:
+
residual_unit_851:


residual_unit_853:

residual_unit_855:

residual_unit_857:

residual_unit_859:

	dense_864:


	dense_866:

identityИв+batch_normalization/StatefulPartitionedCallвconv2d/StatefulPartitionedCallвdense/StatefulPartitionedCallв%residual_unit/StatefulPartitionedCall┘
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs
conv2d_827*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_513Ё
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_830batch_normalization_832batch_normalization_834batch_normalization_836*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_335я
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_531ф
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_355╟
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_841residual_unit_843residual_unit_845residual_unit_847residual_unit_849residual_unit_851residual_unit_853residual_unit_855residual_unit_857residual_unit_859*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_residual_unit_layer_call_and_return_conditional_losses_746¤
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_496▐
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_604ї
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_864	dense_866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_617u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
▌
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Д
╛
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1631

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Ц
B
&__inference_flatten_layer_call_fn_1543

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_604`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╖

ї
,__inference_residual_unit_layer_call_fn_1445

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
identityИвStatefulPartitionedCall╟
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_residual_unit_layer_call_and_return_conditional_losses_746w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         
: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
Р
S
7__inference_global_average_pooling2d_layer_call_fn_1532

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_496i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ>
Е
F__inference_residual_unit_layer_call_and_return_conditional_losses_746

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
identityИв$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1вconv2d_1/Conv2D/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpО
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0л
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0─
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╖
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0─
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:         
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:         
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:         
м
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         
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
:         

 
_user_specified_nameinputs
▒
]
A__inference_flatten_layer_call_and_return_conditional_losses_1549

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
┘
╖
)__inference_sequential_layer_call_fn_1118

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
identityИвStatefulPartitionedCallб
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
:         
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
а'
╙
C__inference_sequential_layer_call_and_return_conditional_losses_624

inputs$

conv2d_514:
%
batch_normalization_517:
%
batch_normalization_519:
%
batch_normalization_521:
%
batch_normalization_523:
+
residual_unit_576:


residual_unit_578:

residual_unit_580:

residual_unit_582:

residual_unit_584:
+
residual_unit_586:


residual_unit_588:

residual_unit_590:

residual_unit_592:

residual_unit_594:

	dense_618:


	dense_620:

identityИв+batch_normalization/StatefulPartitionedCallвconv2d/StatefulPartitionedCallвdense/StatefulPartitionedCallв%residual_unit/StatefulPartitionedCall┘
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs
conv2d_514*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_513Є
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_517batch_normalization_519batch_normalization_521batch_normalization_523*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_304я
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_531ф
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_355╦
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_576residual_unit_578residual_unit_580residual_unit_582residual_unit_584residual_unit_586residual_unit_588residual_unit_590residual_unit_592residual_unit_594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_residual_unit_layer_call_and_return_conditional_losses_575¤
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_496▐
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_604ї
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_618	dense_620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_617u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
▌
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▓'
┘
C__inference_sequential_layer_call_and_return_conditional_losses_992
conv2d_input$

conv2d_949:
%
batch_normalization_952:
%
batch_normalization_954:
%
batch_normalization_956:
%
batch_normalization_958:
+
residual_unit_963:


residual_unit_965:

residual_unit_967:

residual_unit_969:

residual_unit_971:
+
residual_unit_973:


residual_unit_975:

residual_unit_977:

residual_unit_979:

residual_unit_981:

	dense_986:


	dense_988:

identityИв+batch_normalization/StatefulPartitionedCallвconv2d/StatefulPartitionedCallвdense/StatefulPartitionedCallв%residual_unit/StatefulPartitionedCall▀
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input
conv2d_949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_513Є
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_952batch_normalization_954batch_normalization_956batch_normalization_958*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_304я
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_531ф
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_355╦
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_963residual_unit_965residual_unit_967residual_unit_969residual_unit_971residual_unit_973residual_unit_975residual_unit_977residual_unit_979residual_unit_981*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_residual_unit_layer_call_and_return_conditional_losses_575¤
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_496▐
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_604ї
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_986	dense_988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_617u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
▌
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input
К	
╧
4__inference_batch_normalization_2_layer_call_fn_1657

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_475Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Ъ
▒
@__inference_conv2d_layer_call_and_return_conditional_losses_1313

inputs8
conv2d_readvariableop_resource:

identityИвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:         
^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ъ

я
>__inference_dense_layer_call_and_return_conditional_losses_617

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
ш
`
D__inference_activation_layer_call_and_return_conditional_losses_1385

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         
b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
Ы

Ё
?__inference_dense_layer_call_and_return_conditional_losses_1569

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
зI
ч
 __inference__traced_restore_1828
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

identity_18ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9│
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┘
value╧B╠B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHФ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_7AssignVariableOp0assignvariableop_7_residual_unit_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_8AssignVariableOp<assignvariableop_8_residual_unit_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_9AssignVariableOp;assignvariableop_9_residual_unit_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_10AssignVariableOp1assignvariableop_10_residual_unit_conv2d_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_11AssignVariableOp=assignvariableop_11_residual_unit_batch_normalization_2_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_12AssignVariableOp<assignvariableop_12_residual_unit_batch_normalization_2_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_13AssignVariableOpCassignvariableop_13_residual_unit_batch_normalization_1_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_14AssignVariableOpGassignvariableop_14_residual_unit_batch_normalization_1_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_15AssignVariableOpCassignvariableop_15_residual_unit_batch_normalization_2_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_16AssignVariableOpGassignvariableop_16_residual_unit_batch_normalization_2_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ┼
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: ▓
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_18Identity_18:output:0*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
╩
Ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1675

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
ъ>
Ж
G__inference_residual_unit_layer_call_and_return_conditional_losses_1527

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
identityИв$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в5batch_normalization_1/FusedBatchNormV3/ReadVariableOpв7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_1/ReadVariableOpв&batch_normalization_1/ReadVariableOp_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1вconv2d_1/Conv2D/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpО
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0л
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0─
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╖
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0─
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
addAddV2*batch_normalization_2/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:         
Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:         
k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:         
м
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         
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
:         

 
_user_specified_nameinputs
Н
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_355

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▓
m
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_496

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
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦'
щ
D__inference_sequential_layer_call_and_return_conditional_losses_1038
conv2d_input$

conv2d_995:
%
batch_normalization_998:
&
batch_normalization_1000:
&
batch_normalization_1002:
&
batch_normalization_1004:
,
residual_unit_1009:

 
residual_unit_1011:
 
residual_unit_1013:
 
residual_unit_1015:
 
residual_unit_1017:
,
residual_unit_1019:

 
residual_unit_1021:
 
residual_unit_1023:
 
residual_unit_1025:
 
residual_unit_1027:


dense_1032:



dense_1034:

identityИв+batch_normalization/StatefulPartitionedCallвconv2d/StatefulPartitionedCallвdense/StatefulPartitionedCallв%residual_unit/StatefulPartitionedCall▀
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input
conv2d_995*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_513є
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_998batch_normalization_1000batch_normalization_1002batch_normalization_1004*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_335я
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_531ф
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_355╤
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_1009residual_unit_1011residual_unit_1013residual_unit_1015residual_unit_1017residual_unit_1019residual_unit_1021residual_unit_1023residual_unit_1025residual_unit_1027*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_residual_unit_layer_call_and_return_conditional_losses_746¤
(global_average_pooling2d/PartitionedCallPartitionedCall.residual_unit/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_496▐
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_604ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_1032
dense_1034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_617u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
▌
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input
├v
ї
D__inference_sequential_layer_call_and_return_conditional_losses_1299

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
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1вconv2d/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв2residual_unit/batch_normalization_1/AssignNewValueв4residual_unit/batch_normalization_1/AssignNewValue_1вCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвEresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в2residual_unit/batch_normalization_1/ReadVariableOpв4residual_unit/batch_normalization_1/ReadVariableOp_1в2residual_unit/batch_normalization_2/AssignNewValueв4residual_unit/batch_normalization_2/AssignNewValue_1вCresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвEresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в2residual_unit/batch_normalization_2/ReadVariableOpв4residual_unit/batch_normalization_2/ReadVariableOp_1в,residual_unit/conv2d_1/Conv2D/ReadVariableOpв,residual_unit/conv2d_2/Conv2D/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0з
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:
*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:
*
dtype0м
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╕
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ц
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
л
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:         
*
ksize
*
paddingSAME*
strides
к
,residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0▀
residual_unit/conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:04residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
к
2residual_unit/batch_normalization_1/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0о
4residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0╠
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0╨
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ш
4residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_1/Conv2D:output:0:residual_unit/batch_normalization_1/ReadVariableOp:value:0<residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╓
2residual_unit/batch_normalization_1/AssignNewValueAssignVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceAresidual_unit/batch_normalization_1/FusedBatchNormV3:batch_mean:0D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(р
4residual_unit/batch_normalization_1/AssignNewValue_1AssignVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceEresidual_unit/batch_normalization_1/FusedBatchNormV3:batch_variance:0F^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
residual_unit/ReluRelu8residual_unit/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
к
,residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0с
residual_unit/conv2d_2/Conv2DConv2D residual_unit/Relu:activations:04residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
к
2residual_unit/batch_normalization_2/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0о
4residual_unit/batch_normalization_2/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0╠
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0╨
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0Ш
4residual_unit/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_2/Conv2D:output:0:residual_unit/batch_normalization_2/ReadVariableOp:value:0<residual_unit/batch_normalization_2/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╓
2residual_unit/batch_normalization_2/AssignNewValueAssignVariableOpLresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceAresidual_unit/batch_normalization_2/FusedBatchNormV3:batch_mean:0D^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(р
4residual_unit/batch_normalization_2/AssignNewValue_1AssignVariableOpNresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceEresidual_unit/batch_normalization_2/FusedBatchNormV3:batch_variance:0F^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(о
residual_unit/addAddV28residual_unit/batch_normalization_2/FusedBatchNormV3:y:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         
m
residual_unit/Relu_1Reluresidual_unit/add:z:0*
T0*/
_output_shapes
:         
А
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╡
global_average_pooling2d/MeanMean"residual_unit/Relu_1:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   М
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:         
А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:         
f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
╥	
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp3^residual_unit/batch_normalization_1/AssignNewValue5^residual_unit/batch_normalization_1/AssignNewValue_1D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_1/ReadVariableOp5^residual_unit/batch_normalization_1/ReadVariableOp_13^residual_unit/batch_normalization_2/AssignNewValue5^residual_unit/batch_normalization_2/AssignNewValue_1D^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_2/ReadVariableOp5^residual_unit/batch_normalization_2/ReadVariableOp_1-^residual_unit/conv2d_1/Conv2D/ReadVariableOp-^residual_unit/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2H
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
4residual_unit/batch_normalization_1/AssignNewValue_14residual_unit/batch_normalization_1/AssignNewValue_12К
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2О
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_1/ReadVariableOp2residual_unit/batch_normalization_1/ReadVariableOp2l
4residual_unit/batch_normalization_1/ReadVariableOp_14residual_unit/batch_normalization_1/ReadVariableOp_12h
2residual_unit/batch_normalization_2/AssignNewValue2residual_unit/batch_normalization_2/AssignNewValue2l
4residual_unit/batch_normalization_2/AssignNewValue_14residual_unit/batch_normalization_2/AssignNewValue_12К
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2О
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_2/ReadVariableOp2residual_unit/batch_normalization_2/ReadVariableOp2l
4residual_unit/batch_normalization_2/ReadVariableOp_14residual_unit/batch_normalization_2/ReadVariableOp_12\
,residual_unit/conv2d_1/Conv2D/ReadVariableOp,residual_unit/conv2d_1/Conv2D/ReadVariableOp2\
,residual_unit/conv2d_2/Conv2D/ReadVariableOp,residual_unit/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
М	
╧
4__inference_batch_normalization_2_layer_call_fn_1644

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_444Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
╟
Ч
L__inference_batch_normalization_layer_call_and_return_conditional_losses_304

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
╒]
╤
D__inference_sequential_layer_call_and_return_conditional_losses_1228

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
identityИв3batch_normalization/FusedBatchNormV3/ReadVariableOpв5batch_normalization/FusedBatchNormV3/ReadVariableOp_1в"batch_normalization/ReadVariableOpв$batch_normalization/ReadVariableOp_1вconv2d/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвEresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в2residual_unit/batch_normalization_1/ReadVariableOpв4residual_unit/batch_normalization_1/ReadVariableOp_1вCresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвEresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в2residual_unit/batch_normalization_2/ReadVariableOpв4residual_unit/batch_normalization_2/ReadVariableOp_1в,residual_unit/conv2d_1/Conv2D/ReadVariableOpв,residual_unit/conv2d_2/Conv2D/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0з
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:
*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:
*
dtype0м
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0к
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( {
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
л
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:         
*
ksize
*
paddingSAME*
strides
к
,residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0▀
residual_unit/conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:04residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
к
2residual_unit/batch_normalization_1/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0о
4residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0╠
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0╨
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0К
4residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_1/Conv2D:output:0:residual_unit/batch_normalization_1/ReadVariableOp:value:0<residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( О
residual_unit/ReluRelu8residual_unit/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
к
,residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0с
residual_unit/conv2d_2/Conv2DConv2D residual_unit/Relu:activations:04residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
к
2residual_unit/batch_normalization_2/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0о
4residual_unit/batch_normalization_2/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0╠
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0╨
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0К
4residual_unit/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_2/Conv2D:output:0:residual_unit/batch_normalization_2/ReadVariableOp:value:0<residual_unit/batch_normalization_2/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( о
residual_unit/addAddV28residual_unit/batch_normalization_2/FusedBatchNormV3:y:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         
m
residual_unit/Relu_1Reluresidual_unit/add:z:0*
T0*/
_output_shapes
:         
А
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╡
global_average_pooling2d/MeanMean"residual_unit/Relu_1:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   М
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*'
_output_shapes
:         
А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:         
f
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
о
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOpD^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_1/ReadVariableOp5^residual_unit/batch_normalization_1/ReadVariableOp_1D^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_2/ReadVariableOp5^residual_unit/batch_normalization_2/ReadVariableOp_1-^residual_unit/conv2d_1/Conv2D/ReadVariableOp-^residual_unit/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2К
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2О
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_1/ReadVariableOp2residual_unit/batch_normalization_1/ReadVariableOp2l
4residual_unit/batch_normalization_1/ReadVariableOp_14residual_unit/batch_normalization_1/ReadVariableOp_12К
Cresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2О
Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_2/ReadVariableOp2residual_unit/batch_normalization_2/ReadVariableOp2l
4residual_unit/batch_normalization_2/ReadVariableOp_14residual_unit/batch_normalization_2/ReadVariableOp_12\
,residual_unit/conv2d_1/Conv2D/ReadVariableOp,residual_unit/conv2d_1/Conv2D/ReadVariableOp2\
,residual_unit/conv2d_2/Conv2D/ReadVariableOp,residual_unit/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
░
\
@__inference_flatten_layer_call_and_return_conditional_losses_604

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
┐
╢
"__inference_signature_wrapper_1079
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
identityИвStatefulPartitionedCallВ
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
:         
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__wrapped_model_282o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input
К	
╧
4__inference_batch_normalization_1_layer_call_fn_1595

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_411Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
╖
С
$__inference_dense_layer_call_fn_1558

inputs
unknown:


	unknown_0:

identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_617o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╝
E
)__inference_activation_layer_call_fn_1380

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_531h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
╚
Ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1357

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
│
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1538

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
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╖l
з
__inference__wrapped_model_282
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
identityИв>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpв@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1в-sequential/batch_normalization/ReadVariableOpв/sequential/batch_normalization/ReadVariableOp_1в'sequential/conv2d/Conv2D/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpвNsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpвPsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1в=sequential/residual_unit/batch_normalization_1/ReadVariableOpв?sequential/residual_unit/batch_normalization_1/ReadVariableOp_1вNsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвPsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в=sequential/residual_unit/batch_normalization_2/ReadVariableOpв?sequential/residual_unit/batch_normalization_2/ReadVariableOp_1в7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOpв7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOpа
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0├
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
а
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:
*
dtype0д
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:
*
dtype0┬
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0╞
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0ь
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3!sequential/conv2d/Conv2D:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( С
sequential/activation/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
┴
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/activation/Relu:activations:0*/
_output_shapes
:         
*
ksize
*
paddingSAME*
strides
└
7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@sequential_residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0А
(sequential/residual_unit/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:0?sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
└
=sequential/residual_unit/batch_normalization_1/ReadVariableOpReadVariableOpFsequential_residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:
*
dtype0─
?sequential/residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOpHsequential_residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:
*
dtype0т
Nsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWsequential_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0ц
Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsequential_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╠
?sequential/residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV31sequential/residual_unit/conv2d_1/Conv2D:output:0Esequential/residual_unit/batch_normalization_1/ReadVariableOp:value:0Gsequential/residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Vsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Xsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( д
sequential/residual_unit/ReluReluCsequential/residual_unit/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         
└
7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@sequential_residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0В
(sequential/residual_unit/conv2d_2/Conv2DConv2D+sequential/residual_unit/Relu:activations:0?sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
└
=sequential/residual_unit/batch_normalization_2/ReadVariableOpReadVariableOpFsequential_residual_unit_batch_normalization_2_readvariableop_resource*
_output_shapes
:
*
dtype0─
?sequential/residual_unit/batch_normalization_2/ReadVariableOp_1ReadVariableOpHsequential_residual_unit_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:
*
dtype0т
Nsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWsequential_residual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0ц
Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYsequential_residual_unit_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╠
?sequential/residual_unit/batch_normalization_2/FusedBatchNormV3FusedBatchNormV31sequential/residual_unit/conv2d_2/Conv2D:output:0Esequential/residual_unit/batch_normalization_2/ReadVariableOp:value:0Gsequential/residual_unit/batch_normalization_2/ReadVariableOp_1:value:0Vsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Xsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         
:
:
:
:
:*
epsilon%oГ:*
is_training( ╧
sequential/residual_unit/addAddV2Csequential/residual_unit/batch_normalization_2/FusedBatchNormV3:y:0)sequential/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         
Г
sequential/residual_unit/Relu_1Relu sequential/residual_unit/add:z:0*
T0*/
_output_shapes
:         
Л
:sequential/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╓
(sequential/global_average_pooling2d/MeanMean-sequential/residual_unit/Relu_1:activations:0Csequential/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   н
sequential/flatten/ReshapeReshape1sequential/global_average_pooling2d/Mean:output:0!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:         
Ц
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0и
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
x
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         
q
IdentityIdentity"sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
щ
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOpO^sequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpQ^sequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1>^sequential/residual_unit/batch_normalization_1/ReadVariableOp@^sequential/residual_unit/batch_normalization_1/ReadVariableOp_1O^sequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpQ^sequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1>^sequential/residual_unit/batch_normalization_2/ReadVariableOp@^sequential/residual_unit/batch_normalization_2/ReadVariableOp_18^sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp8^sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 2А
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2Д
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2а
Nsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpNsequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2д
Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Psequential/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12~
=sequential/residual_unit/batch_normalization_1/ReadVariableOp=sequential/residual_unit/batch_normalization_1/ReadVariableOp2В
?sequential/residual_unit/batch_normalization_1/ReadVariableOp_1?sequential/residual_unit/batch_normalization_1/ReadVariableOp_12а
Nsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOpNsequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2д
Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Psequential/residual_unit/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12~
=sequential/residual_unit/batch_normalization_2/ReadVariableOp=sequential/residual_unit/batch_normalization_2/ReadVariableOp2В
?sequential/residual_unit/batch_normalization_2/ReadVariableOp_1?sequential/residual_unit/batch_normalization_2/ReadVariableOp_12r
7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp7sequential/residual_unit/conv2d_1/Conv2D/ReadVariableOp2r
7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp7sequential/residual_unit/conv2d_2/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input
ъ
╝
(__inference_sequential_layer_call_fn_661
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
identityИвStatefulPartitionedCallз
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
:         
*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input
О
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1395

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Б
╗
L__inference_batch_normalization_layer_call_and_return_conditional_losses_335

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
Щ
░
?__inference_conv2d_layer_call_and_return_conditional_losses_513

inputs8
conv2d_readvariableop_resource:

identityИвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:         
^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ф
╝
(__inference_sequential_layer_call_fn_946
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
identityИвStatefulPartitionedCallб
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
:         
*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input
╔
Щ
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_444

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs
╣
Б
%__inference_conv2d_layer_call_fn_1306

inputs!
unknown:

identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_513w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╔
Щ
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_380

inputs%
readvariableop_resource:
'
readvariableop_1_resource:
6
(fusedbatchnormv3_readvariableop_resource:
8
*fusedbatchnormv3_readvariableop_1_resource:

identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:
*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:
*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:
*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           
:
:
:
:
:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           
░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           

 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*║
serving_defaultж
M
conv2d_input=
serving_default_conv2d_input:0         9
dense0
StatefulPartitionedCall:0         
tensorflow/serving/predict:ме
з
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

signatures"
_tf_keras_sequential
╙
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 _jit_compiled_convolution_op"
_tf_keras_layer
ъ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	 gamma
!beta
"moving_mean
#moving_variance"
_tf_keras_layer
е
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
е
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
╟
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6main_layers
7skip_layers"
_tf_keras_layer
е
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
е
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias"
_tf_keras_layer
Ю
0
 1
!2
"3
#4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
J15
K16"
trackable_list_wrapper
n
0
 1
!2
L3
M4
N5
O6
P7
Q8
J9
K10"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╫
[trace_0
\trace_1
]trace_2
^trace_32ь
(__inference_sequential_layer_call_fn_661
)__inference_sequential_layer_call_fn_1118
)__inference_sequential_layer_call_fn_1157
(__inference_sequential_layer_call_fn_946┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z[trace_0z\trace_1z]trace_2z^trace_3
─
_trace_0
`trace_1
atrace_2
btrace_32┘
D__inference_sequential_layer_call_and_return_conditional_losses_1228
D__inference_sequential_layer_call_and_return_conditional_losses_1299
C__inference_sequential_layer_call_and_return_conditional_losses_992
D__inference_sequential_layer_call_and_return_conditional_losses_1038┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z_trace_0z`trace_1zatrace_2zbtrace_3
╬B╦
__inference__wrapped_model_282conv2d_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
cserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
н
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
щ
itrace_02╠
%__inference_conv2d_layer_call_fn_1306в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zitrace_0
Д
jtrace_02ч
@__inference_conv2d_layer_call_and_return_conditional_losses_1313в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zjtrace_0
':%
2conv2d/kernel
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
 0
!1
"2
#3"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╒
ptrace_0
qtrace_12Ю
2__inference_batch_normalization_layer_call_fn_1326
2__inference_batch_normalization_layer_call_fn_1339│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zptrace_0zqtrace_1
Л
rtrace_0
strace_12╘
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1357
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1375│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zrtrace_0zstrace_1
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
н
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
э
ytrace_02╨
)__inference_activation_layer_call_fn_1380в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zytrace_0
И
ztrace_02ы
D__inference_activation_layer_call_and_return_conditional_losses_1385в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
Є
Аtrace_02╙
,__inference_max_pooling2d_layer_call_fn_1390в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zАtrace_0
Н
Бtrace_02ю
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1395в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zБtrace_0
f
L0
M1
N2
O3
P4
Q5
R6
S7
T8
U9"
trackable_list_wrapper
J
L0
M1
N2
O3
P4
Q5"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
┘
Зtrace_0
Иtrace_12Ю
,__inference_residual_unit_layer_call_fn_1420
,__inference_residual_unit_layer_call_fn_1445┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zЗtrace_0zИtrace_1
П
Йtrace_0
Кtrace_12╘
G__inference_residual_unit_layer_call_and_return_conditional_losses_1486
G__inference_residual_unit_layer_call_and_return_conditional_losses_1527┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 zЙtrace_0zКtrace_1
@
Л0
М1
Н3
О4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
¤
Фtrace_02▐
7__inference_global_average_pooling2d_layer_call_fn_1532в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
Ш
Хtrace_02∙
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1538в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zХtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
ь
Ыtrace_02═
&__inference_flatten_layer_call_fn_1543в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
З
Ьtrace_02ш
A__inference_flatten_layer_call_and_return_conditional_losses_1549в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ъ
вtrace_02╦
$__inference_dense_layer_call_fn_1558в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
Е
гtrace_02ц
?__inference_dense_layer_call_and_return_conditional_losses_1569в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zгtrace_0
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
"0
#1
R2
S3
T4
U5"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 B№
(__inference_sequential_layer_call_fn_661conv2d_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
)__inference_sequential_layer_call_fn_1118inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
)__inference_sequential_layer_call_fn_1157inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
(__inference_sequential_layer_call_fn_946conv2d_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
D__inference_sequential_layer_call_and_return_conditional_losses_1228inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
D__inference_sequential_layer_call_and_return_conditional_losses_1299inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
C__inference_sequential_layer_call_and_return_conditional_losses_992conv2d_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЫBШ
D__inference_sequential_layer_call_and_return_conditional_losses_1038conv2d_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬B╦
"__inference_signature_wrapper_1079conv2d_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┘B╓
%__inference_conv2d_layer_call_fn_1306inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_conv2d_layer_call_and_return_conditional_losses_1313inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ўBЇ
2__inference_batch_normalization_layer_call_fn_1326inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
2__inference_batch_normalization_layer_call_fn_1339inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1357inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1375inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▌B┌
)__inference_activation_layer_call_fn_1380inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_activation_layer_call_and_return_conditional_losses_1385inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_max_pooling2d_layer_call_fn_1390inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1395inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
<
R0
S1
T2
U3"
trackable_list_wrapper
@
Л0
М1
Н2
О3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¤B·
,__inference_residual_unit_layer_call_fn_1420inputs"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
¤B·
,__inference_residual_unit_layer_call_fn_1445inputs"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ШBХ
G__inference_residual_unit_layer_call_and_return_conditional_losses_1486inputs"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ШBХ
G__inference_residual_unit_layer_call_and_return_conditional_losses_1527inputs"┐
╢▓▓
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
┌
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses

Lkernel
!к_jit_compiled_convolution_op"
_tf_keras_layer
ё
л	variables
мtrainable_variables
нregularization_losses
о	keras_api
п__call__
+░&call_and_return_all_conditional_losses
	▒axis
	Mgamma
Nbeta
Rmoving_mean
Smoving_variance"
_tf_keras_layer
┌
▓	variables
│trainable_variables
┤regularization_losses
╡	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses

Okernel
!╕_jit_compiled_convolution_op"
_tf_keras_layer
ё
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses
	┐axis
	Pgamma
Qbeta
Tmoving_mean
Umoving_variance"
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
ыBш
7__inference_global_average_pooling2d_layer_call_fn_1532inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1538inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┌B╫
&__inference_flatten_layer_call_fn_1543inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_flatten_layer_call_and_return_conditional_losses_1549inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╪B╒
$__inference_dense_layer_call_fn_1558inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
?__inference_dense_layer_call_and_return_conditional_losses_1569inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
'
L0"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
M0
N1
R2
S3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
л	variables
мtrainable_variables
нregularization_losses
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
▌
╩trace_0
╦trace_12в
4__inference_batch_normalization_1_layer_call_fn_1582
4__inference_batch_normalization_1_layer_call_fn_1595│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0z╦trace_1
У
╠trace_0
═trace_12╪
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1613
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1631│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0z═trace_1
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
P0
Q1
T2
U3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
╣	variables
║trainable_variables
╗regularization_losses
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
▌
╪trace_0
┘trace_12в
4__inference_batch_normalization_2_layer_call_fn_1644
4__inference_batch_normalization_2_layer_call_fn_1657│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0z┘trace_1
У
┌trace_0
█trace_12╪
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1675
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1693│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0z█trace_1
 "
trackable_list_wrapper
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
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
∙BЎ
4__inference_batch_normalization_1_layer_call_fn_1582inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
4__inference_batch_normalization_1_layer_call_fn_1595inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1613inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1631inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
∙BЎ
4__inference_batch_normalization_2_layer_call_fn_1644inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
4__inference_batch_normalization_2_layer_call_fn_1657inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1675inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1693inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 д
__inference__wrapped_model_282Б !"#LMNRSOPQTUJK=в:
3в0
.К+
conv2d_input         
к "-к*
(
denseК
dense         
░
D__inference_activation_layer_call_and_return_conditional_losses_1385h7в4
-в*
(К%
inputs         

к "-в*
#К 
0         

Ъ И
)__inference_activation_layer_call_fn_1380[7в4
-в*
(К%
inputs         

к " К         
ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1613ЦMNRSMвJ
Cв@
:К7
inputs+                           

p 
к "?в<
5К2
0+                           

Ъ ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1631ЦMNRSMвJ
Cв@
:К7
inputs+                           

p
к "?в<
5К2
0+                           

Ъ ┬
4__inference_batch_normalization_1_layer_call_fn_1582ЙMNRSMвJ
Cв@
:К7
inputs+                           

p 
к "2К/+                           
┬
4__inference_batch_normalization_1_layer_call_fn_1595ЙMNRSMвJ
Cв@
:К7
inputs+                           

p
к "2К/+                           
ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1675ЦPQTUMвJ
Cв@
:К7
inputs+                           

p 
к "?в<
5К2
0+                           

Ъ ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1693ЦPQTUMвJ
Cв@
:К7
inputs+                           

p
к "?в<
5К2
0+                           

Ъ ┬
4__inference_batch_normalization_2_layer_call_fn_1644ЙPQTUMвJ
Cв@
:К7
inputs+                           

p 
к "2К/+                           
┬
4__inference_batch_normalization_2_layer_call_fn_1657ЙPQTUMвJ
Cв@
:К7
inputs+                           

p
к "2К/+                           
ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1357Ц !"#MвJ
Cв@
:К7
inputs+                           

p 
к "?в<
5К2
0+                           

Ъ ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1375Ц !"#MвJ
Cв@
:К7
inputs+                           

p
к "?в<
5К2
0+                           

Ъ └
2__inference_batch_normalization_layer_call_fn_1326Й !"#MвJ
Cв@
:К7
inputs+                           

p 
к "2К/+                           
└
2__inference_batch_normalization_layer_call_fn_1339Й !"#MвJ
Cв@
:К7
inputs+                           

p
к "2К/+                           
п
@__inference_conv2d_layer_call_and_return_conditional_losses_1313k7в4
-в*
(К%
inputs         
к "-в*
#К 
0         

Ъ З
%__inference_conv2d_layer_call_fn_1306^7в4
-в*
(К%
inputs         
к " К         
Я
?__inference_dense_layer_call_and_return_conditional_losses_1569\JK/в,
%в"
 К
inputs         

к "%в"
К
0         

Ъ w
$__inference_dense_layer_call_fn_1558OJK/в,
%в"
 К
inputs         

к "К         
Э
A__inference_flatten_layer_call_and_return_conditional_losses_1549X/в,
%в"
 К
inputs         

к "%в"
К
0         

Ъ u
&__inference_flatten_layer_call_fn_1543K/в,
%в"
 К
inputs         

к "К         
█
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1538ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ▓
7__inference_global_average_pooling2d_layer_call_fn_1532wRвO
HвE
CК@
inputs4                                    
к "!К                  ъ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1395ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┬
,__inference_max_pooling2d_layer_call_fn_1390СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╨
G__inference_residual_unit_layer_call_and_return_conditional_losses_1486Д
LMNRSOPQTUGвD
-в*
(К%
inputs         

к

trainingp "-в*
#К 
0         

Ъ ╨
G__inference_residual_unit_layer_call_and_return_conditional_losses_1527Д
LMNRSOPQTUGвD
-в*
(К%
inputs         

к

trainingp"-в*
#К 
0         

Ъ з
,__inference_residual_unit_layer_call_fn_1420w
LMNRSOPQTUGвD
-в*
(К%
inputs         

к

trainingp " К         
з
,__inference_residual_unit_layer_call_fn_1445w
LMNRSOPQTUGвD
-в*
(К%
inputs         

к

trainingp" К         
╩
D__inference_sequential_layer_call_and_return_conditional_losses_1038Б !"#LMNRSOPQTUJKEвB
;в8
.К+
conv2d_input         
p

 
к "%в"
К
0         

Ъ ├
D__inference_sequential_layer_call_and_return_conditional_losses_1228{ !"#LMNRSOPQTUJK?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         

Ъ ├
D__inference_sequential_layer_call_and_return_conditional_losses_1299{ !"#LMNRSOPQTUJK?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         

Ъ ╔
C__inference_sequential_layer_call_and_return_conditional_losses_992Б !"#LMNRSOPQTUJKEвB
;в8
.К+
conv2d_input         
p 

 
к "%в"
К
0         

Ъ Ы
)__inference_sequential_layer_call_fn_1118n !"#LMNRSOPQTUJK?в<
5в2
(К%
inputs         
p 

 
к "К         
Ы
)__inference_sequential_layer_call_fn_1157n !"#LMNRSOPQTUJK?в<
5в2
(К%
inputs         
p

 
к "К         
а
(__inference_sequential_layer_call_fn_661t !"#LMNRSOPQTUJKEвB
;в8
.К+
conv2d_input         
p 

 
к "К         
а
(__inference_sequential_layer_call_fn_946t !"#LMNRSOPQTUJKEвB
;в8
.К+
conv2d_input         
p

 
к "К         
╕
"__inference_signature_wrapper_1079С !"#LMNRSOPQTUJKMвJ
в 
Cк@
>
conv2d_input.К+
conv2d_input         "-к*
(
denseК
dense         
