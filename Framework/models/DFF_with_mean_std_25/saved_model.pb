¯×
í¾
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
¾
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
executor_typestring 
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
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8ß

{
dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	D* 
shared_namedense_70/kernel
t
#dense_70/kernel/Read/ReadVariableOpReadVariableOpdense_70/kernel*
_output_shapes
:	D*
dtype0
s
dense_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_70/bias
l
!dense_70/bias/Read/ReadVariableOpReadVariableOpdense_70/bias*
_output_shapes	
:*
dtype0
|
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_71/kernel
u
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel* 
_output_shapes
:
*
dtype0
s
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_71/bias
l
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes	
:*
dtype0
|
dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_72/kernel
u
#dense_72/kernel/Read/ReadVariableOpReadVariableOpdense_72/kernel* 
_output_shapes
:
*
dtype0
s
dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_72/bias
l
!dense_72/bias/Read/ReadVariableOpReadVariableOpdense_72/bias*
_output_shapes	
:*
dtype0
|
dense_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_73/kernel
u
#dense_73/kernel/Read/ReadVariableOpReadVariableOpdense_73/kernel* 
_output_shapes
:
*
dtype0
s
dense_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_73/bias
l
!dense_73/bias/Read/ReadVariableOpReadVariableOpdense_73/bias*
_output_shapes	
:*
dtype0
|
dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_74/kernel
u
#dense_74/kernel/Read/ReadVariableOpReadVariableOpdense_74/kernel* 
_output_shapes
:
*
dtype0
s
dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_74/bias
l
!dense_74/bias/Read/ReadVariableOpReadVariableOpdense_74/bias*
_output_shapes	
:*
dtype0
|
dense_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_75/kernel
u
#dense_75/kernel/Read/ReadVariableOpReadVariableOpdense_75/kernel* 
_output_shapes
:
*
dtype0
s
dense_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_75/bias
l
!dense_75/bias/Read/ReadVariableOpReadVariableOpdense_75/bias*
_output_shapes	
:*
dtype0
|
dense_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_76/kernel
u
#dense_76/kernel/Read/ReadVariableOpReadVariableOpdense_76/kernel* 
_output_shapes
:
*
dtype0
s
dense_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_76/bias
l
!dense_76/bias/Read/ReadVariableOpReadVariableOpdense_76/bias*
_output_shapes	
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	+*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	+*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:+*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
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
n
accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator
g
accumulator/Read/ReadVariableOpReadVariableOpaccumulator*
_output_shapes
:*
dtype0
r
accumulator_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_1
k
!accumulator_1/Read/ReadVariableOpReadVariableOpaccumulator_1*
_output_shapes
:*
dtype0
r
accumulator_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_2
k
!accumulator_2/Read/ReadVariableOpReadVariableOpaccumulator_2*
_output_shapes
:*
dtype0
r
accumulator_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaccumulator_3
k
!accumulator_3/Read/ReadVariableOpReadVariableOpaccumulator_3*
_output_shapes
:*
dtype0

Nadam/dense_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	D*(
shared_nameNadam/dense_70/kernel/m

+Nadam/dense_70/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_70/kernel/m*
_output_shapes
:	D*
dtype0

Nadam/dense_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_70/bias/m
|
)Nadam/dense_70/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_70/bias/m*
_output_shapes	
:*
dtype0

Nadam/dense_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_71/kernel/m

+Nadam/dense_71/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_71/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/dense_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_71/bias/m
|
)Nadam/dense_71/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_71/bias/m*
_output_shapes	
:*
dtype0

Nadam/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_72/kernel/m

+Nadam/dense_72/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_72/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_72/bias/m
|
)Nadam/dense_72/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_72/bias/m*
_output_shapes	
:*
dtype0

Nadam/dense_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_73/kernel/m

+Nadam/dense_73/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_73/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/dense_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_73/bias/m
|
)Nadam/dense_73/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_73/bias/m*
_output_shapes	
:*
dtype0

Nadam/dense_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_74/kernel/m

+Nadam/dense_74/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_74/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/dense_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_74/bias/m
|
)Nadam/dense_74/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_74/bias/m*
_output_shapes	
:*
dtype0

Nadam/dense_75/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_75/kernel/m

+Nadam/dense_75/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_75/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/dense_75/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_75/bias/m
|
)Nadam/dense_75/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_75/bias/m*
_output_shapes	
:*
dtype0

Nadam/dense_76/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_76/kernel/m

+Nadam/dense_76/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_76/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/dense_76/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_76/bias/m
|
)Nadam/dense_76/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_76/bias/m*
_output_shapes	
:*
dtype0

Nadam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	+*&
shared_nameNadam/output/kernel/m

)Nadam/output/kernel/m/Read/ReadVariableOpReadVariableOpNadam/output/kernel/m*
_output_shapes
:	+*
dtype0
~
Nadam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*$
shared_nameNadam/output/bias/m
w
'Nadam/output/bias/m/Read/ReadVariableOpReadVariableOpNadam/output/bias/m*
_output_shapes
:+*
dtype0

Nadam/dense_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	D*(
shared_nameNadam/dense_70/kernel/v

+Nadam/dense_70/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_70/kernel/v*
_output_shapes
:	D*
dtype0

Nadam/dense_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_70/bias/v
|
)Nadam/dense_70/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_70/bias/v*
_output_shapes	
:*
dtype0

Nadam/dense_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_71/kernel/v

+Nadam/dense_71/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_71/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/dense_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_71/bias/v
|
)Nadam/dense_71/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_71/bias/v*
_output_shapes	
:*
dtype0

Nadam/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_72/kernel/v

+Nadam/dense_72/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_72/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_72/bias/v
|
)Nadam/dense_72/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_72/bias/v*
_output_shapes	
:*
dtype0

Nadam/dense_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_73/kernel/v

+Nadam/dense_73/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_73/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/dense_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_73/bias/v
|
)Nadam/dense_73/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_73/bias/v*
_output_shapes	
:*
dtype0

Nadam/dense_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_74/kernel/v

+Nadam/dense_74/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_74/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/dense_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_74/bias/v
|
)Nadam/dense_74/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_74/bias/v*
_output_shapes	
:*
dtype0

Nadam/dense_75/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_75/kernel/v

+Nadam/dense_75/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_75/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/dense_75/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_75/bias/v
|
)Nadam/dense_75/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_75/bias/v*
_output_shapes	
:*
dtype0

Nadam/dense_76/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameNadam/dense_76/kernel/v

+Nadam/dense_76/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_76/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/dense_76/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_76/bias/v
|
)Nadam/dense_76/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_76/bias/v*
_output_shapes	
:*
dtype0

Nadam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	+*&
shared_nameNadam/output/kernel/v

)Nadam/output/kernel/v/Read/ReadVariableOpReadVariableOpNadam/output/kernel/v*
_output_shapes
:	+*
dtype0
~
Nadam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:+*$
shared_nameNadam/output/bias/v
w
'Nadam/output/bias/v/Read/ReadVariableOpReadVariableOpNadam/output/bias/v*
_output_shapes
:+*
dtype0

NoOpNoOp
ùX
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*´X
valueªXB§X B X
¶
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api

@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate
Emomentum_cachemmmmmm"m#m(m)m.m/m4m5m:m ;m¡v¢v£v¤v¥v¦v§"v¨#v©(vª)v«.v¬/v­4v®5v¯:v°;v±
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
v
0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15
 
­
Flayer_metrics
Glayer_regularization_losses
	variables
trainable_variables
Hnon_trainable_variables

Ilayers
regularization_losses
Jmetrics
 
[Y
VARIABLE_VALUEdense_70/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_70/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Klayer_metrics
Llayer_regularization_losses
	variables
trainable_variables
Mnon_trainable_variables

Nlayers
regularization_losses
Ometrics
[Y
VARIABLE_VALUEdense_71/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_71/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Player_metrics
Qlayer_regularization_losses
	variables
trainable_variables
Rnon_trainable_variables

Slayers
regularization_losses
Tmetrics
[Y
VARIABLE_VALUEdense_72/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_72/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Ulayer_metrics
Vlayer_regularization_losses
	variables
trainable_variables
Wnon_trainable_variables

Xlayers
 regularization_losses
Ymetrics
[Y
VARIABLE_VALUEdense_73/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_73/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
­
Zlayer_metrics
[layer_regularization_losses
$	variables
%trainable_variables
\non_trainable_variables

]layers
&regularization_losses
^metrics
[Y
VARIABLE_VALUEdense_74/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_74/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
­
_layer_metrics
`layer_regularization_losses
*	variables
+trainable_variables
anon_trainable_variables

blayers
,regularization_losses
cmetrics
[Y
VARIABLE_VALUEdense_75/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_75/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
­
dlayer_metrics
elayer_regularization_losses
0	variables
1trainable_variables
fnon_trainable_variables

glayers
2regularization_losses
hmetrics
[Y
VARIABLE_VALUEdense_76/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_76/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
­
ilayer_metrics
jlayer_regularization_losses
6	variables
7trainable_variables
knon_trainable_variables

llayers
8regularization_losses
mmetrics
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
­
nlayer_metrics
olayer_regularization_losses
<	variables
=trainable_variables
pnon_trainable_variables

qlayers
>regularization_losses
rmetrics
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
?
0
1
2
3
4
5
6
7
	8
*
s0
t1
u2
v3
w4
x5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ytotal
	zcount
{	variables
|	keras_api
F
	}total
	~count

_fn_kwargs
	variables
	keras_api
C

thresholds
accumulator
	variables
	keras_api
C

thresholds
accumulator
	variables
	keras_api
C

thresholds
accumulator
	variables
	keras_api
C

thresholds
accumulator
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

{	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

	variables
 
[Y
VARIABLE_VALUEaccumulator:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0

	variables
 
][
VARIABLE_VALUEaccumulator_1:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0

	variables
 
][
VARIABLE_VALUEaccumulator_2:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0

	variables
 
][
VARIABLE_VALUEaccumulator_3:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUE

0

	variables
}
VARIABLE_VALUENadam/dense_70/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_70/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_71/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_71/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_72/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_72/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_73/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_73/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_74/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_74/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_75/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_75/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_76/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_76/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/output/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/output/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_70/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_70/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_71/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_71/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_72/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_72/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_73/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_73/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_74/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_74/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_75/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_75/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_76/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_76/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/output/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/output/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_first_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿD
Ò
StatefulPartitionedCallStatefulPartitionedCallserving_default_first_inputdense_70/kerneldense_70/biasdense_71/kerneldense_71/biasdense_72/kerneldense_72/biasdense_73/kerneldense_73/biasdense_74/kerneldense_74/biasdense_75/kerneldense_75/biasdense_76/kerneldense_76/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_688569
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ß
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_70/kernel/Read/ReadVariableOp!dense_70/bias/Read/ReadVariableOp#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOp#dense_72/kernel/Read/ReadVariableOp!dense_72/bias/Read/ReadVariableOp#dense_73/kernel/Read/ReadVariableOp!dense_73/bias/Read/ReadVariableOp#dense_74/kernel/Read/ReadVariableOp!dense_74/bias/Read/ReadVariableOp#dense_75/kernel/Read/ReadVariableOp!dense_75/bias/Read/ReadVariableOp#dense_76/kernel/Read/ReadVariableOp!dense_76/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpaccumulator/Read/ReadVariableOp!accumulator_1/Read/ReadVariableOp!accumulator_2/Read/ReadVariableOp!accumulator_3/Read/ReadVariableOp+Nadam/dense_70/kernel/m/Read/ReadVariableOp)Nadam/dense_70/bias/m/Read/ReadVariableOp+Nadam/dense_71/kernel/m/Read/ReadVariableOp)Nadam/dense_71/bias/m/Read/ReadVariableOp+Nadam/dense_72/kernel/m/Read/ReadVariableOp)Nadam/dense_72/bias/m/Read/ReadVariableOp+Nadam/dense_73/kernel/m/Read/ReadVariableOp)Nadam/dense_73/bias/m/Read/ReadVariableOp+Nadam/dense_74/kernel/m/Read/ReadVariableOp)Nadam/dense_74/bias/m/Read/ReadVariableOp+Nadam/dense_75/kernel/m/Read/ReadVariableOp)Nadam/dense_75/bias/m/Read/ReadVariableOp+Nadam/dense_76/kernel/m/Read/ReadVariableOp)Nadam/dense_76/bias/m/Read/ReadVariableOp)Nadam/output/kernel/m/Read/ReadVariableOp'Nadam/output/bias/m/Read/ReadVariableOp+Nadam/dense_70/kernel/v/Read/ReadVariableOp)Nadam/dense_70/bias/v/Read/ReadVariableOp+Nadam/dense_71/kernel/v/Read/ReadVariableOp)Nadam/dense_71/bias/v/Read/ReadVariableOp+Nadam/dense_72/kernel/v/Read/ReadVariableOp)Nadam/dense_72/bias/v/Read/ReadVariableOp+Nadam/dense_73/kernel/v/Read/ReadVariableOp)Nadam/dense_73/bias/v/Read/ReadVariableOp+Nadam/dense_74/kernel/v/Read/ReadVariableOp)Nadam/dense_74/bias/v/Read/ReadVariableOp+Nadam/dense_75/kernel/v/Read/ReadVariableOp)Nadam/dense_75/bias/v/Read/ReadVariableOp+Nadam/dense_76/kernel/v/Read/ReadVariableOp)Nadam/dense_76/bias/v/Read/ReadVariableOp)Nadam/output/kernel/v/Read/ReadVariableOp'Nadam/output/bias/v/Read/ReadVariableOpConst*K
TinD
B2@	*
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
__inference__traced_save_689132

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_70/kerneldense_70/biasdense_71/kerneldense_71/biasdense_72/kerneldense_72/biasdense_73/kerneldense_73/biasdense_74/kerneldense_74/biasdense_75/kerneldense_75/biasdense_76/kerneldense_76/biasoutput/kerneloutput/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1accumulatoraccumulator_1accumulator_2accumulator_3Nadam/dense_70/kernel/mNadam/dense_70/bias/mNadam/dense_71/kernel/mNadam/dense_71/bias/mNadam/dense_72/kernel/mNadam/dense_72/bias/mNadam/dense_73/kernel/mNadam/dense_73/bias/mNadam/dense_74/kernel/mNadam/dense_74/bias/mNadam/dense_75/kernel/mNadam/dense_75/bias/mNadam/dense_76/kernel/mNadam/dense_76/bias/mNadam/output/kernel/mNadam/output/bias/mNadam/dense_70/kernel/vNadam/dense_70/bias/vNadam/dense_71/kernel/vNadam/dense_71/bias/vNadam/dense_72/kernel/vNadam/dense_72/bias/vNadam/dense_73/kernel/vNadam/dense_73/bias/vNadam/dense_74/kernel/vNadam/dense_74/bias/vNadam/dense_75/kernel/vNadam/dense_75/bias/vNadam/dense_76/kernel/vNadam/dense_76/bias/vNadam/output/kernel/vNadam/output/bias/v*J
TinC
A2?*
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
"__inference__traced_restore_689328ó
÷	
Ý
D__inference_dense_73_layer_call_and_return_conditional_losses_688190

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_76_layer_call_and_return_conditional_losses_688894

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
*
Ê
D__inference_model_10_layer_call_and_return_conditional_losses_688315
first_input
dense_70_688120
dense_70_688122
dense_71_688147
dense_71_688149
dense_72_688174
dense_72_688176
dense_73_688201
dense_73_688203
dense_74_688228
dense_74_688230
dense_75_688255
dense_75_688257
dense_76_688282
dense_76_688284
output_688309
output_688311
identity¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢ dense_75/StatefulPartitionedCall¢ dense_76/StatefulPartitionedCall¢output/StatefulPartitionedCall
 dense_70/StatefulPartitionedCallStatefulPartitionedCallfirst_inputdense_70_688120dense_70_688122*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_6881092"
 dense_70/StatefulPartitionedCall¸
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_688147dense_71_688149*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6881362"
 dense_71/StatefulPartitionedCall¸
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_688174dense_72_688176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_6881632"
 dense_72/StatefulPartitionedCall¸
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_688201dense_73_688203*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_6881902"
 dense_73/StatefulPartitionedCall¸
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_688228dense_74_688230*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_6882172"
 dense_74/StatefulPartitionedCall¸
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_688255dense_75_688257*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_6882442"
 dense_75/StatefulPartitionedCall¸
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_688282dense_76_688284*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_76_layer_call_and_return_conditional_losses_6882712"
 dense_76/StatefulPartitionedCall­
output/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0output_688309output_688311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6882982 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
%
_user_specified_namefirst_input
Þ
~
)__inference_dense_70_layer_call_fn_688783

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_6881092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
ô	
Ý
D__inference_dense_70_layer_call_and_return_conditional_losses_688774

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	D*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_71_layer_call_and_return_conditional_losses_688794

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_74_layer_call_and_return_conditional_losses_688217

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
~
)__inference_dense_73_layer_call_fn_688843

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_6881902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶

Ú
)__inference_model_10_layer_call_fn_688441
first_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_6884062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
%
_user_specified_namefirst_input
§

Õ
)__inference_model_10_layer_call_fn_688726

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_6884062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
*
Ê
D__inference_model_10_layer_call_and_return_conditional_losses_688359
first_input
dense_70_688318
dense_70_688320
dense_71_688323
dense_71_688325
dense_72_688328
dense_72_688330
dense_73_688333
dense_73_688335
dense_74_688338
dense_74_688340
dense_75_688343
dense_75_688345
dense_76_688348
dense_76_688350
output_688353
output_688355
identity¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢ dense_75/StatefulPartitionedCall¢ dense_76/StatefulPartitionedCall¢output/StatefulPartitionedCall
 dense_70/StatefulPartitionedCallStatefulPartitionedCallfirst_inputdense_70_688318dense_70_688320*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_6881092"
 dense_70/StatefulPartitionedCall¸
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_688323dense_71_688325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6881362"
 dense_71/StatefulPartitionedCall¸
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_688328dense_72_688330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_6881632"
 dense_72/StatefulPartitionedCall¸
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_688333dense_73_688335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_6881902"
 dense_73/StatefulPartitionedCall¸
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_688338dense_74_688340*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_6882172"
 dense_74/StatefulPartitionedCall¸
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_688343dense_75_688345*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_6882442"
 dense_75/StatefulPartitionedCall¸
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_688348dense_76_688350*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_76_layer_call_and_return_conditional_losses_6882712"
 dense_76/StatefulPartitionedCall­
output/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0output_688353output_688355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6882982 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
%
_user_specified_namefirst_input
÷	
Û
B__inference_output_layer_call_and_return_conditional_losses_688298

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_73_layer_call_and_return_conditional_losses_688834

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Õ
$__inference_signature_wrapper_688569
first_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_6880942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
%
_user_specified_namefirst_input
÷	
Ý
D__inference_dense_76_layer_call_and_return_conditional_losses_688271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Û
B__inference_output_layer_call_and_return_conditional_losses_688914

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	+*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_72_layer_call_and_return_conditional_losses_688814

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶

Ú
)__inference_model_10_layer_call_fn_688522
first_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallfirst_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_6884872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
%
_user_specified_namefirst_input
à
~
)__inference_dense_74_layer_call_fn_688863

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_6882172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
|
'__inference_output_layer_call_fn_688923

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6882982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

Õ
)__inference_model_10_layer_call_fn_688763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_10_layer_call_and_return_conditional_losses_6884872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_75_layer_call_and_return_conditional_losses_688244

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_71_layer_call_and_return_conditional_losses_688136

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
K
Ë

D__inference_model_10_layer_call_and_return_conditional_losses_688689

inputs+
'dense_70_matmul_readvariableop_resource,
(dense_70_biasadd_readvariableop_resource+
'dense_71_matmul_readvariableop_resource,
(dense_71_biasadd_readvariableop_resource+
'dense_72_matmul_readvariableop_resource,
(dense_72_biasadd_readvariableop_resource+
'dense_73_matmul_readvariableop_resource,
(dense_73_biasadd_readvariableop_resource+
'dense_74_matmul_readvariableop_resource,
(dense_74_biasadd_readvariableop_resource+
'dense_75_matmul_readvariableop_resource,
(dense_75_biasadd_readvariableop_resource+
'dense_76_matmul_readvariableop_resource,
(dense_76_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢dense_72/BiasAdd/ReadVariableOp¢dense_72/MatMul/ReadVariableOp¢dense_73/BiasAdd/ReadVariableOp¢dense_73/MatMul/ReadVariableOp¢dense_74/BiasAdd/ReadVariableOp¢dense_74/MatMul/ReadVariableOp¢dense_75/BiasAdd/ReadVariableOp¢dense_75/MatMul/ReadVariableOp¢dense_76/BiasAdd/ReadVariableOp¢dense_76/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp©
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes
:	D*
dtype02 
dense_70/MatMul/ReadVariableOp
dense_70/MatMulMatMulinputs&dense_70/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_70/MatMul¨
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_70/BiasAdd/ReadVariableOp¦
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_70/BiasAddt
dense_70/ReluReludense_70/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_70/Reluª
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_71/MatMul/ReadVariableOp¤
dense_71/MatMulMatMuldense_70/Relu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/MatMul¨
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_71/BiasAdd/ReadVariableOp¦
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/BiasAddt
dense_71/ReluReludense_71/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/Reluª
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_72/MatMul/ReadVariableOp¤
dense_72/MatMulMatMuldense_71/Relu:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/MatMul¨
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_72/BiasAdd/ReadVariableOp¦
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/BiasAddt
dense_72/ReluReludense_72/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/Reluª
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_73/MatMul/ReadVariableOp¤
dense_73/MatMulMatMuldense_72/Relu:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_73/MatMul¨
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_73/BiasAdd/ReadVariableOp¦
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_73/BiasAddt
dense_73/ReluReludense_73/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_73/Reluª
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_74/MatMul/ReadVariableOp¤
dense_74/MatMulMatMuldense_73/Relu:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_74/MatMul¨
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_74/BiasAdd/ReadVariableOp¦
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_74/BiasAddt
dense_74/ReluReludense_74/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_74/Reluª
dense_75/MatMul/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_75/MatMul/ReadVariableOp¤
dense_75/MatMulMatMuldense_74/Relu:activations:0&dense_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_75/MatMul¨
dense_75/BiasAdd/ReadVariableOpReadVariableOp(dense_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_75/BiasAdd/ReadVariableOp¦
dense_75/BiasAddBiasAdddense_75/MatMul:product:0'dense_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_75/BiasAddt
dense_75/ReluReludense_75/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_75/Reluª
dense_76/MatMul/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_76/MatMul/ReadVariableOp¤
dense_76/MatMulMatMuldense_75/Relu:activations:0&dense_76/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_76/MatMul¨
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_76/BiasAdd/ReadVariableOp¦
dense_76/BiasAddBiasAdddense_76/MatMul:product:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_76/BiasAddt
dense_76/ReluReludense_76/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_76/Relu£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	+*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense_76/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
output/Softmax
IdentityIdentityoutput/Softmax:softmax:0 ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp ^dense_75/BiasAdd/ReadVariableOp^dense_75/MatMul/ReadVariableOp ^dense_76/BiasAdd/ReadVariableOp^dense_76/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2B
dense_75/BiasAdd/ReadVariableOpdense_75/BiasAdd/ReadVariableOp2@
dense_75/MatMul/ReadVariableOpdense_75/MatMul/ReadVariableOp2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2@
dense_76/MatMul/ReadVariableOpdense_76/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_72_layer_call_and_return_conditional_losses_688163

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
~
)__inference_dense_75_layer_call_fn_688883

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_6882442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
~
)__inference_dense_76_layer_call_fn_688903

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_76_layer_call_and_return_conditional_losses_6882712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

 
"__inference__traced_restore_689328
file_prefix$
 assignvariableop_dense_70_kernel$
 assignvariableop_1_dense_70_bias&
"assignvariableop_2_dense_71_kernel$
 assignvariableop_3_dense_71_bias&
"assignvariableop_4_dense_72_kernel$
 assignvariableop_5_dense_72_bias&
"assignvariableop_6_dense_73_kernel$
 assignvariableop_7_dense_73_bias&
"assignvariableop_8_dense_74_kernel$
 assignvariableop_9_dense_74_bias'
#assignvariableop_10_dense_75_kernel%
!assignvariableop_11_dense_75_bias'
#assignvariableop_12_dense_76_kernel%
!assignvariableop_13_dense_76_bias%
!assignvariableop_14_output_kernel#
assignvariableop_15_output_bias"
assignvariableop_16_nadam_iter$
 assignvariableop_17_nadam_beta_1$
 assignvariableop_18_nadam_beta_2#
assignvariableop_19_nadam_decay+
'assignvariableop_20_nadam_learning_rate,
(assignvariableop_21_nadam_momentum_cache
assignvariableop_22_total
assignvariableop_23_count
assignvariableop_24_total_1
assignvariableop_25_count_1#
assignvariableop_26_accumulator%
!assignvariableop_27_accumulator_1%
!assignvariableop_28_accumulator_2%
!assignvariableop_29_accumulator_3/
+assignvariableop_30_nadam_dense_70_kernel_m-
)assignvariableop_31_nadam_dense_70_bias_m/
+assignvariableop_32_nadam_dense_71_kernel_m-
)assignvariableop_33_nadam_dense_71_bias_m/
+assignvariableop_34_nadam_dense_72_kernel_m-
)assignvariableop_35_nadam_dense_72_bias_m/
+assignvariableop_36_nadam_dense_73_kernel_m-
)assignvariableop_37_nadam_dense_73_bias_m/
+assignvariableop_38_nadam_dense_74_kernel_m-
)assignvariableop_39_nadam_dense_74_bias_m/
+assignvariableop_40_nadam_dense_75_kernel_m-
)assignvariableop_41_nadam_dense_75_bias_m/
+assignvariableop_42_nadam_dense_76_kernel_m-
)assignvariableop_43_nadam_dense_76_bias_m-
)assignvariableop_44_nadam_output_kernel_m+
'assignvariableop_45_nadam_output_bias_m/
+assignvariableop_46_nadam_dense_70_kernel_v-
)assignvariableop_47_nadam_dense_70_bias_v/
+assignvariableop_48_nadam_dense_71_kernel_v-
)assignvariableop_49_nadam_dense_71_bias_v/
+assignvariableop_50_nadam_dense_72_kernel_v-
)assignvariableop_51_nadam_dense_72_bias_v/
+assignvariableop_52_nadam_dense_73_kernel_v-
)assignvariableop_53_nadam_dense_73_bias_v/
+assignvariableop_54_nadam_dense_74_kernel_v-
)assignvariableop_55_nadam_dense_74_bias_v/
+assignvariableop_56_nadam_dense_75_kernel_v-
)assignvariableop_57_nadam_dense_75_bias_v/
+assignvariableop_58_nadam_dense_76_kernel_v-
)assignvariableop_59_nadam_dense_76_bias_v-
)assignvariableop_60_nadam_output_kernel_v+
'assignvariableop_61_nadam_output_bias_v
identity_63¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ù"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*å!
valueÛ!BØ!?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesé
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesÿ
ü:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_70_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_70_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_71_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_71_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_72_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_72_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_73_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_73_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_74_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_74_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_75_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_75_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_76_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_76_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15§
AssignVariableOp_15AssignVariableOpassignvariableop_15_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¦
AssignVariableOp_16AssignVariableOpassignvariableop_16_nadam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_nadam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¨
AssignVariableOp_18AssignVariableOp assignvariableop_18_nadam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_nadam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_nadam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21°
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_momentum_cacheIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24£
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26§
AssignVariableOp_26AssignVariableOpassignvariableop_26_accumulatorIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27©
AssignVariableOp_27AssignVariableOp!assignvariableop_27_accumulator_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28©
AssignVariableOp_28AssignVariableOp!assignvariableop_28_accumulator_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29©
AssignVariableOp_29AssignVariableOp!assignvariableop_29_accumulator_3Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30³
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_70_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_70_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32³
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_71_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_71_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34³
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_dense_72_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35±
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_dense_72_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36³
AssignVariableOp_36AssignVariableOp+assignvariableop_36_nadam_dense_73_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_nadam_dense_73_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38³
AssignVariableOp_38AssignVariableOp+assignvariableop_38_nadam_dense_74_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_nadam_dense_74_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40³
AssignVariableOp_40AssignVariableOp+assignvariableop_40_nadam_dense_75_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_nadam_dense_75_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42³
AssignVariableOp_42AssignVariableOp+assignvariableop_42_nadam_dense_76_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43±
AssignVariableOp_43AssignVariableOp)assignvariableop_43_nadam_dense_76_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_nadam_output_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¯
AssignVariableOp_45AssignVariableOp'assignvariableop_45_nadam_output_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46³
AssignVariableOp_46AssignVariableOp+assignvariableop_46_nadam_dense_70_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47±
AssignVariableOp_47AssignVariableOp)assignvariableop_47_nadam_dense_70_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48³
AssignVariableOp_48AssignVariableOp+assignvariableop_48_nadam_dense_71_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49±
AssignVariableOp_49AssignVariableOp)assignvariableop_49_nadam_dense_71_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50³
AssignVariableOp_50AssignVariableOp+assignvariableop_50_nadam_dense_72_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51±
AssignVariableOp_51AssignVariableOp)assignvariableop_51_nadam_dense_72_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52³
AssignVariableOp_52AssignVariableOp+assignvariableop_52_nadam_dense_73_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53±
AssignVariableOp_53AssignVariableOp)assignvariableop_53_nadam_dense_73_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54³
AssignVariableOp_54AssignVariableOp+assignvariableop_54_nadam_dense_74_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55±
AssignVariableOp_55AssignVariableOp)assignvariableop_55_nadam_dense_74_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56³
AssignVariableOp_56AssignVariableOp+assignvariableop_56_nadam_dense_75_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57±
AssignVariableOp_57AssignVariableOp)assignvariableop_57_nadam_dense_75_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58³
AssignVariableOp_58AssignVariableOp+assignvariableop_58_nadam_dense_76_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59±
AssignVariableOp_59AssignVariableOp)assignvariableop_59_nadam_dense_76_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60±
AssignVariableOp_60AssignVariableOp)assignvariableop_60_nadam_output_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¯
AssignVariableOp_61AssignVariableOp'assignvariableop_61_nadam_output_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_619
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp²
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_62¥
Identity_63IdentityIdentity_62:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_63"#
identity_63Identity_63:output:0*
_input_shapesý
ú: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
÷	
Ý
D__inference_dense_74_layer_call_and_return_conditional_losses_688854

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
*
Å
D__inference_model_10_layer_call_and_return_conditional_losses_688487

inputs
dense_70_688446
dense_70_688448
dense_71_688451
dense_71_688453
dense_72_688456
dense_72_688458
dense_73_688461
dense_73_688463
dense_74_688466
dense_74_688468
dense_75_688471
dense_75_688473
dense_76_688476
dense_76_688478
output_688481
output_688483
identity¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢ dense_75/StatefulPartitionedCall¢ dense_76/StatefulPartitionedCall¢output/StatefulPartitionedCall
 dense_70/StatefulPartitionedCallStatefulPartitionedCallinputsdense_70_688446dense_70_688448*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_6881092"
 dense_70/StatefulPartitionedCall¸
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_688451dense_71_688453*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6881362"
 dense_71/StatefulPartitionedCall¸
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_688456dense_72_688458*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_6881632"
 dense_72/StatefulPartitionedCall¸
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_688461dense_73_688463*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_6881902"
 dense_73/StatefulPartitionedCall¸
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_688466dense_74_688468*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_6882172"
 dense_74/StatefulPartitionedCall¸
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_688471dense_75_688473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_6882442"
 dense_75/StatefulPartitionedCall¸
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_688476dense_76_688478*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_76_layer_call_and_return_conditional_losses_6882712"
 dense_76/StatefulPartitionedCall­
output/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0output_688481output_688483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6882982 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
à
~
)__inference_dense_71_layer_call_fn_688803

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6881362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_75_layer_call_and_return_conditional_losses_688874

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
K
Ë

D__inference_model_10_layer_call_and_return_conditional_losses_688629

inputs+
'dense_70_matmul_readvariableop_resource,
(dense_70_biasadd_readvariableop_resource+
'dense_71_matmul_readvariableop_resource,
(dense_71_biasadd_readvariableop_resource+
'dense_72_matmul_readvariableop_resource,
(dense_72_biasadd_readvariableop_resource+
'dense_73_matmul_readvariableop_resource,
(dense_73_biasadd_readvariableop_resource+
'dense_74_matmul_readvariableop_resource,
(dense_74_biasadd_readvariableop_resource+
'dense_75_matmul_readvariableop_resource,
(dense_75_biasadd_readvariableop_resource+
'dense_76_matmul_readvariableop_resource,
(dense_76_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢dense_72/BiasAdd/ReadVariableOp¢dense_72/MatMul/ReadVariableOp¢dense_73/BiasAdd/ReadVariableOp¢dense_73/MatMul/ReadVariableOp¢dense_74/BiasAdd/ReadVariableOp¢dense_74/MatMul/ReadVariableOp¢dense_75/BiasAdd/ReadVariableOp¢dense_75/MatMul/ReadVariableOp¢dense_76/BiasAdd/ReadVariableOp¢dense_76/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp©
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes
:	D*
dtype02 
dense_70/MatMul/ReadVariableOp
dense_70/MatMulMatMulinputs&dense_70/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_70/MatMul¨
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_70/BiasAdd/ReadVariableOp¦
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_70/BiasAddt
dense_70/ReluReludense_70/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_70/Reluª
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_71/MatMul/ReadVariableOp¤
dense_71/MatMulMatMuldense_70/Relu:activations:0&dense_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/MatMul¨
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_71/BiasAdd/ReadVariableOp¦
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/BiasAddt
dense_71/ReluReludense_71/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_71/Reluª
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_72/MatMul/ReadVariableOp¤
dense_72/MatMulMatMuldense_71/Relu:activations:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/MatMul¨
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_72/BiasAdd/ReadVariableOp¦
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/BiasAddt
dense_72/ReluReludense_72/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_72/Reluª
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_73/MatMul/ReadVariableOp¤
dense_73/MatMulMatMuldense_72/Relu:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_73/MatMul¨
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_73/BiasAdd/ReadVariableOp¦
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_73/BiasAddt
dense_73/ReluReludense_73/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_73/Reluª
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_74/MatMul/ReadVariableOp¤
dense_74/MatMulMatMuldense_73/Relu:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_74/MatMul¨
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_74/BiasAdd/ReadVariableOp¦
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_74/BiasAddt
dense_74/ReluReludense_74/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_74/Reluª
dense_75/MatMul/ReadVariableOpReadVariableOp'dense_75_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_75/MatMul/ReadVariableOp¤
dense_75/MatMulMatMuldense_74/Relu:activations:0&dense_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_75/MatMul¨
dense_75/BiasAdd/ReadVariableOpReadVariableOp(dense_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_75/BiasAdd/ReadVariableOp¦
dense_75/BiasAddBiasAdddense_75/MatMul:product:0'dense_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_75/BiasAddt
dense_75/ReluReludense_75/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_75/Reluª
dense_76/MatMul/ReadVariableOpReadVariableOp'dense_76_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_76/MatMul/ReadVariableOp¤
dense_76/MatMulMatMuldense_75/Relu:activations:0&dense_76/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_76/MatMul¨
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_76/BiasAdd/ReadVariableOp¦
dense_76/BiasAddBiasAdddense_76/MatMul:product:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_76/BiasAddt
dense_76/ReluReludense_76/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_76/Relu£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	+*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense_76/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
output/Softmax
IdentityIdentityoutput/Softmax:softmax:0 ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp ^dense_75/BiasAdd/ReadVariableOp^dense_75/MatMul/ReadVariableOp ^dense_76/BiasAdd/ReadVariableOp^dense_76/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2B
dense_75/BiasAdd/ReadVariableOpdense_75/BiasAdd/ReadVariableOp2@
dense_75/MatMul/ReadVariableOpdense_75/MatMul/ReadVariableOp2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2@
dense_76/MatMul/ReadVariableOpdense_76/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
*
Å
D__inference_model_10_layer_call_and_return_conditional_losses_688406

inputs
dense_70_688365
dense_70_688367
dense_71_688370
dense_71_688372
dense_72_688375
dense_72_688377
dense_73_688380
dense_73_688382
dense_74_688385
dense_74_688387
dense_75_688390
dense_75_688392
dense_76_688395
dense_76_688397
output_688400
output_688402
identity¢ dense_70/StatefulPartitionedCall¢ dense_71/StatefulPartitionedCall¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢ dense_75/StatefulPartitionedCall¢ dense_76/StatefulPartitionedCall¢output/StatefulPartitionedCall
 dense_70/StatefulPartitionedCallStatefulPartitionedCallinputsdense_70_688365dense_70_688367*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_70_layer_call_and_return_conditional_losses_6881092"
 dense_70/StatefulPartitionedCall¸
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_688370dense_71_688372*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_71_layer_call_and_return_conditional_losses_6881362"
 dense_71/StatefulPartitionedCall¸
 dense_72/StatefulPartitionedCallStatefulPartitionedCall)dense_71/StatefulPartitionedCall:output:0dense_72_688375dense_72_688377*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_6881632"
 dense_72/StatefulPartitionedCall¸
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0dense_73_688380dense_73_688382*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_6881902"
 dense_73/StatefulPartitionedCall¸
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_688385dense_74_688387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_6882172"
 dense_74/StatefulPartitionedCall¸
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_688390dense_75_688392*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_6882442"
 dense_75/StatefulPartitionedCall¸
 dense_76/StatefulPartitionedCallStatefulPartitionedCall)dense_75/StatefulPartitionedCall:output:0dense_76_688395dense_76_688397*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_76_layer_call_and_return_conditional_losses_6882712"
 dense_76/StatefulPartitionedCall­
output/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0output_688400output_688402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_6882982 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall!^dense_71/StatefulPartitionedCall!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall!^dense_76/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
ô	
Ý
D__inference_dense_70_layer_call_and_return_conditional_losses_688109

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	D*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿD::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
 
_user_specified_nameinputs
Z
Í
!__inference__wrapped_model_688094
first_input4
0model_10_dense_70_matmul_readvariableop_resource5
1model_10_dense_70_biasadd_readvariableop_resource4
0model_10_dense_71_matmul_readvariableop_resource5
1model_10_dense_71_biasadd_readvariableop_resource4
0model_10_dense_72_matmul_readvariableop_resource5
1model_10_dense_72_biasadd_readvariableop_resource4
0model_10_dense_73_matmul_readvariableop_resource5
1model_10_dense_73_biasadd_readvariableop_resource4
0model_10_dense_74_matmul_readvariableop_resource5
1model_10_dense_74_biasadd_readvariableop_resource4
0model_10_dense_75_matmul_readvariableop_resource5
1model_10_dense_75_biasadd_readvariableop_resource4
0model_10_dense_76_matmul_readvariableop_resource5
1model_10_dense_76_biasadd_readvariableop_resource2
.model_10_output_matmul_readvariableop_resource3
/model_10_output_biasadd_readvariableop_resource
identity¢(model_10/dense_70/BiasAdd/ReadVariableOp¢'model_10/dense_70/MatMul/ReadVariableOp¢(model_10/dense_71/BiasAdd/ReadVariableOp¢'model_10/dense_71/MatMul/ReadVariableOp¢(model_10/dense_72/BiasAdd/ReadVariableOp¢'model_10/dense_72/MatMul/ReadVariableOp¢(model_10/dense_73/BiasAdd/ReadVariableOp¢'model_10/dense_73/MatMul/ReadVariableOp¢(model_10/dense_74/BiasAdd/ReadVariableOp¢'model_10/dense_74/MatMul/ReadVariableOp¢(model_10/dense_75/BiasAdd/ReadVariableOp¢'model_10/dense_75/MatMul/ReadVariableOp¢(model_10/dense_76/BiasAdd/ReadVariableOp¢'model_10/dense_76/MatMul/ReadVariableOp¢&model_10/output/BiasAdd/ReadVariableOp¢%model_10/output/MatMul/ReadVariableOpÄ
'model_10/dense_70/MatMul/ReadVariableOpReadVariableOp0model_10_dense_70_matmul_readvariableop_resource*
_output_shapes
:	D*
dtype02)
'model_10/dense_70/MatMul/ReadVariableOp¯
model_10/dense_70/MatMulMatMulfirst_input/model_10/dense_70/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_70/MatMulÃ
(model_10/dense_70/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_70_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_70/BiasAdd/ReadVariableOpÊ
model_10/dense_70/BiasAddBiasAdd"model_10/dense_70/MatMul:product:00model_10/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_70/BiasAdd
model_10/dense_70/ReluRelu"model_10/dense_70/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_70/ReluÅ
'model_10/dense_71/MatMul/ReadVariableOpReadVariableOp0model_10_dense_71_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_10/dense_71/MatMul/ReadVariableOpÈ
model_10/dense_71/MatMulMatMul$model_10/dense_70/Relu:activations:0/model_10/dense_71/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_71/MatMulÃ
(model_10/dense_71/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_71_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_71/BiasAdd/ReadVariableOpÊ
model_10/dense_71/BiasAddBiasAdd"model_10/dense_71/MatMul:product:00model_10/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_71/BiasAdd
model_10/dense_71/ReluRelu"model_10/dense_71/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_71/ReluÅ
'model_10/dense_72/MatMul/ReadVariableOpReadVariableOp0model_10_dense_72_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_10/dense_72/MatMul/ReadVariableOpÈ
model_10/dense_72/MatMulMatMul$model_10/dense_71/Relu:activations:0/model_10/dense_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_72/MatMulÃ
(model_10/dense_72/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_72_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_72/BiasAdd/ReadVariableOpÊ
model_10/dense_72/BiasAddBiasAdd"model_10/dense_72/MatMul:product:00model_10/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_72/BiasAdd
model_10/dense_72/ReluRelu"model_10/dense_72/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_72/ReluÅ
'model_10/dense_73/MatMul/ReadVariableOpReadVariableOp0model_10_dense_73_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_10/dense_73/MatMul/ReadVariableOpÈ
model_10/dense_73/MatMulMatMul$model_10/dense_72/Relu:activations:0/model_10/dense_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_73/MatMulÃ
(model_10/dense_73/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_73_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_73/BiasAdd/ReadVariableOpÊ
model_10/dense_73/BiasAddBiasAdd"model_10/dense_73/MatMul:product:00model_10/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_73/BiasAdd
model_10/dense_73/ReluRelu"model_10/dense_73/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_73/ReluÅ
'model_10/dense_74/MatMul/ReadVariableOpReadVariableOp0model_10_dense_74_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_10/dense_74/MatMul/ReadVariableOpÈ
model_10/dense_74/MatMulMatMul$model_10/dense_73/Relu:activations:0/model_10/dense_74/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_74/MatMulÃ
(model_10/dense_74/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_74_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_74/BiasAdd/ReadVariableOpÊ
model_10/dense_74/BiasAddBiasAdd"model_10/dense_74/MatMul:product:00model_10/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_74/BiasAdd
model_10/dense_74/ReluRelu"model_10/dense_74/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_74/ReluÅ
'model_10/dense_75/MatMul/ReadVariableOpReadVariableOp0model_10_dense_75_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_10/dense_75/MatMul/ReadVariableOpÈ
model_10/dense_75/MatMulMatMul$model_10/dense_74/Relu:activations:0/model_10/dense_75/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_75/MatMulÃ
(model_10/dense_75/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_75_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_75/BiasAdd/ReadVariableOpÊ
model_10/dense_75/BiasAddBiasAdd"model_10/dense_75/MatMul:product:00model_10/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_75/BiasAdd
model_10/dense_75/ReluRelu"model_10/dense_75/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_75/ReluÅ
'model_10/dense_76/MatMul/ReadVariableOpReadVariableOp0model_10_dense_76_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_10/dense_76/MatMul/ReadVariableOpÈ
model_10/dense_76/MatMulMatMul$model_10/dense_75/Relu:activations:0/model_10/dense_76/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_76/MatMulÃ
(model_10/dense_76/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_76_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_10/dense_76/BiasAdd/ReadVariableOpÊ
model_10/dense_76/BiasAddBiasAdd"model_10/dense_76/MatMul:product:00model_10/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_76/BiasAdd
model_10/dense_76/ReluRelu"model_10/dense_76/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_10/dense_76/Relu¾
%model_10/output/MatMul/ReadVariableOpReadVariableOp.model_10_output_matmul_readvariableop_resource*
_output_shapes
:	+*
dtype02'
%model_10/output/MatMul/ReadVariableOpÁ
model_10/output/MatMulMatMul$model_10/dense_76/Relu:activations:0-model_10/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
model_10/output/MatMul¼
&model_10/output/BiasAdd/ReadVariableOpReadVariableOp/model_10_output_biasadd_readvariableop_resource*
_output_shapes
:+*
dtype02(
&model_10/output/BiasAdd/ReadVariableOpÁ
model_10/output/BiasAddBiasAdd model_10/output/MatMul:product:0.model_10/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
model_10/output/BiasAdd
model_10/output/SoftmaxSoftmax model_10/output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2
model_10/output/Softmax
IdentityIdentity!model_10/output/Softmax:softmax:0)^model_10/dense_70/BiasAdd/ReadVariableOp(^model_10/dense_70/MatMul/ReadVariableOp)^model_10/dense_71/BiasAdd/ReadVariableOp(^model_10/dense_71/MatMul/ReadVariableOp)^model_10/dense_72/BiasAdd/ReadVariableOp(^model_10/dense_72/MatMul/ReadVariableOp)^model_10/dense_73/BiasAdd/ReadVariableOp(^model_10/dense_73/MatMul/ReadVariableOp)^model_10/dense_74/BiasAdd/ReadVariableOp(^model_10/dense_74/MatMul/ReadVariableOp)^model_10/dense_75/BiasAdd/ReadVariableOp(^model_10/dense_75/MatMul/ReadVariableOp)^model_10/dense_76/BiasAdd/ReadVariableOp(^model_10/dense_76/MatMul/ReadVariableOp'^model_10/output/BiasAdd/ReadVariableOp&^model_10/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ+2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿD::::::::::::::::2T
(model_10/dense_70/BiasAdd/ReadVariableOp(model_10/dense_70/BiasAdd/ReadVariableOp2R
'model_10/dense_70/MatMul/ReadVariableOp'model_10/dense_70/MatMul/ReadVariableOp2T
(model_10/dense_71/BiasAdd/ReadVariableOp(model_10/dense_71/BiasAdd/ReadVariableOp2R
'model_10/dense_71/MatMul/ReadVariableOp'model_10/dense_71/MatMul/ReadVariableOp2T
(model_10/dense_72/BiasAdd/ReadVariableOp(model_10/dense_72/BiasAdd/ReadVariableOp2R
'model_10/dense_72/MatMul/ReadVariableOp'model_10/dense_72/MatMul/ReadVariableOp2T
(model_10/dense_73/BiasAdd/ReadVariableOp(model_10/dense_73/BiasAdd/ReadVariableOp2R
'model_10/dense_73/MatMul/ReadVariableOp'model_10/dense_73/MatMul/ReadVariableOp2T
(model_10/dense_74/BiasAdd/ReadVariableOp(model_10/dense_74/BiasAdd/ReadVariableOp2R
'model_10/dense_74/MatMul/ReadVariableOp'model_10/dense_74/MatMul/ReadVariableOp2T
(model_10/dense_75/BiasAdd/ReadVariableOp(model_10/dense_75/BiasAdd/ReadVariableOp2R
'model_10/dense_75/MatMul/ReadVariableOp'model_10/dense_75/MatMul/ReadVariableOp2T
(model_10/dense_76/BiasAdd/ReadVariableOp(model_10/dense_76/BiasAdd/ReadVariableOp2R
'model_10/dense_76/MatMul/ReadVariableOp'model_10/dense_76/MatMul/ReadVariableOp2P
&model_10/output/BiasAdd/ReadVariableOp&model_10/output/BiasAdd/ReadVariableOp2N
%model_10/output/MatMul/ReadVariableOp%model_10/output/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
%
_user_specified_namefirst_input
Ñy

__inference__traced_save_689132
file_prefix.
*savev2_dense_70_kernel_read_readvariableop,
(savev2_dense_70_bias_read_readvariableop.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop.
*savev2_dense_72_kernel_read_readvariableop,
(savev2_dense_72_bias_read_readvariableop.
*savev2_dense_73_kernel_read_readvariableop,
(savev2_dense_73_bias_read_readvariableop.
*savev2_dense_74_kernel_read_readvariableop,
(savev2_dense_74_bias_read_readvariableop.
*savev2_dense_75_kernel_read_readvariableop,
(savev2_dense_75_bias_read_readvariableop.
*savev2_dense_76_kernel_read_readvariableop,
(savev2_dense_76_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop*
&savev2_accumulator_read_readvariableop,
(savev2_accumulator_1_read_readvariableop,
(savev2_accumulator_2_read_readvariableop,
(savev2_accumulator_3_read_readvariableop6
2savev2_nadam_dense_70_kernel_m_read_readvariableop4
0savev2_nadam_dense_70_bias_m_read_readvariableop6
2savev2_nadam_dense_71_kernel_m_read_readvariableop4
0savev2_nadam_dense_71_bias_m_read_readvariableop6
2savev2_nadam_dense_72_kernel_m_read_readvariableop4
0savev2_nadam_dense_72_bias_m_read_readvariableop6
2savev2_nadam_dense_73_kernel_m_read_readvariableop4
0savev2_nadam_dense_73_bias_m_read_readvariableop6
2savev2_nadam_dense_74_kernel_m_read_readvariableop4
0savev2_nadam_dense_74_bias_m_read_readvariableop6
2savev2_nadam_dense_75_kernel_m_read_readvariableop4
0savev2_nadam_dense_75_bias_m_read_readvariableop6
2savev2_nadam_dense_76_kernel_m_read_readvariableop4
0savev2_nadam_dense_76_bias_m_read_readvariableop4
0savev2_nadam_output_kernel_m_read_readvariableop2
.savev2_nadam_output_bias_m_read_readvariableop6
2savev2_nadam_dense_70_kernel_v_read_readvariableop4
0savev2_nadam_dense_70_bias_v_read_readvariableop6
2savev2_nadam_dense_71_kernel_v_read_readvariableop4
0savev2_nadam_dense_71_bias_v_read_readvariableop6
2savev2_nadam_dense_72_kernel_v_read_readvariableop4
0savev2_nadam_dense_72_bias_v_read_readvariableop6
2savev2_nadam_dense_73_kernel_v_read_readvariableop4
0savev2_nadam_dense_73_bias_v_read_readvariableop6
2savev2_nadam_dense_74_kernel_v_read_readvariableop4
0savev2_nadam_dense_74_bias_v_read_readvariableop6
2savev2_nadam_dense_75_kernel_v_read_readvariableop4
0savev2_nadam_dense_75_bias_v_read_readvariableop6
2savev2_nadam_dense_76_kernel_v_read_readvariableop4
0savev2_nadam_dense_76_bias_v_read_readvariableop4
0savev2_nadam_output_kernel_v_read_readvariableop2
.savev2_nadam_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÓ"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*å!
valueÛ!BØ!?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/2/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/3/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/4/accumulator/.ATTRIBUTES/VARIABLE_VALUEB:keras_api/metrics/5/accumulator/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices°
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_70_kernel_read_readvariableop(savev2_dense_70_bias_read_readvariableop*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableop*savev2_dense_72_kernel_read_readvariableop(savev2_dense_72_bias_read_readvariableop*savev2_dense_73_kernel_read_readvariableop(savev2_dense_73_bias_read_readvariableop*savev2_dense_74_kernel_read_readvariableop(savev2_dense_74_bias_read_readvariableop*savev2_dense_75_kernel_read_readvariableop(savev2_dense_75_bias_read_readvariableop*savev2_dense_76_kernel_read_readvariableop(savev2_dense_76_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop&savev2_accumulator_read_readvariableop(savev2_accumulator_1_read_readvariableop(savev2_accumulator_2_read_readvariableop(savev2_accumulator_3_read_readvariableop2savev2_nadam_dense_70_kernel_m_read_readvariableop0savev2_nadam_dense_70_bias_m_read_readvariableop2savev2_nadam_dense_71_kernel_m_read_readvariableop0savev2_nadam_dense_71_bias_m_read_readvariableop2savev2_nadam_dense_72_kernel_m_read_readvariableop0savev2_nadam_dense_72_bias_m_read_readvariableop2savev2_nadam_dense_73_kernel_m_read_readvariableop0savev2_nadam_dense_73_bias_m_read_readvariableop2savev2_nadam_dense_74_kernel_m_read_readvariableop0savev2_nadam_dense_74_bias_m_read_readvariableop2savev2_nadam_dense_75_kernel_m_read_readvariableop0savev2_nadam_dense_75_bias_m_read_readvariableop2savev2_nadam_dense_76_kernel_m_read_readvariableop0savev2_nadam_dense_76_bias_m_read_readvariableop0savev2_nadam_output_kernel_m_read_readvariableop.savev2_nadam_output_bias_m_read_readvariableop2savev2_nadam_dense_70_kernel_v_read_readvariableop0savev2_nadam_dense_70_bias_v_read_readvariableop2savev2_nadam_dense_71_kernel_v_read_readvariableop0savev2_nadam_dense_71_bias_v_read_readvariableop2savev2_nadam_dense_72_kernel_v_read_readvariableop0savev2_nadam_dense_72_bias_v_read_readvariableop2savev2_nadam_dense_73_kernel_v_read_readvariableop0savev2_nadam_dense_73_bias_v_read_readvariableop2savev2_nadam_dense_74_kernel_v_read_readvariableop0savev2_nadam_dense_74_bias_v_read_readvariableop2savev2_nadam_dense_75_kernel_v_read_readvariableop0savev2_nadam_dense_75_bias_v_read_readvariableop2savev2_nadam_dense_76_kernel_v_read_readvariableop0savev2_nadam_dense_76_bias_v_read_readvariableop0savev2_nadam_output_kernel_v_read_readvariableop.savev2_nadam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesò
ï: :	D::
::
::
::
::
::
::	+:+: : : : : : : : : : :::::	D::
::
::
::
::
::
::	+:+:	D::
::
::
::
::
::
::	+:+: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	D:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	+: 

_output_shapes
:+:

_output_shapes
: :
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
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	D:! 

_output_shapes	
::&!"
 
_output_shapes
:
:!"

_output_shapes	
::&#"
 
_output_shapes
:
:!$

_output_shapes	
::&%"
 
_output_shapes
:
:!&

_output_shapes	
::&'"
 
_output_shapes
:
:!(

_output_shapes	
::&)"
 
_output_shapes
:
:!*

_output_shapes	
::&+"
 
_output_shapes
:
:!,

_output_shapes	
::%-!

_output_shapes
:	+: .

_output_shapes
:+:%/!

_output_shapes
:	D:!0

_output_shapes	
::&1"
 
_output_shapes
:
:!2

_output_shapes	
::&3"
 
_output_shapes
:
:!4

_output_shapes	
::&5"
 
_output_shapes
:
:!6

_output_shapes	
::&7"
 
_output_shapes
:
:!8

_output_shapes	
::&9"
 
_output_shapes
:
:!:

_output_shapes	
::&;"
 
_output_shapes
:
:!<

_output_shapes	
::%=!

_output_shapes
:	+: >

_output_shapes
:+:?

_output_shapes
: 
à
~
)__inference_dense_72_layer_call_fn_688823

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_6881632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
C
first_input4
serving_default_first_input:0ÿÿÿÿÿÿÿÿÿD:
output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ+tensorflow/serving/predict:õ®
øR
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+²&call_and_return_all_conditional_losses
³_default_save_signature
´__call__"åN
_tf_keras_networkÉN{"class_name": "Functional", "name": "model_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 68]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "first_input"}, "name": "first_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_70", "inbound_nodes": [[["first_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_71", "inbound_nodes": [[["dense_70", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_72", "inbound_nodes": [[["dense_71", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_73", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_73", "inbound_nodes": [[["dense_72", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_74", "inbound_nodes": [[["dense_73", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_75", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_75", "inbound_nodes": [[["dense_74", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_76", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_76", "inbound_nodes": [[["dense_75", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_76", 0, 0, {}]]]}], "input_layers": [["first_input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 68]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 68]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 68]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "first_input"}, "name": "first_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_70", "inbound_nodes": [[["first_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_71", "inbound_nodes": [[["dense_70", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_72", "inbound_nodes": [[["dense_71", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_73", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_73", "inbound_nodes": [[["dense_72", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_74", "inbound_nodes": [[["dense_73", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_75", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_75", "inbound_nodes": [[["dense_74", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_76", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_76", "inbound_nodes": [[["dense_75", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_76", 0, 0, {}]]]}], "input_layers": [["first_input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}, {"class_name": "TruePositives", "config": {"name": "true_positives_10", "dtype": "float32", "thresholds": null}}, {"class_name": "TrueNegatives", "config": {"name": "true_negatives_10", "dtype": "float32", "thresholds": null}}, {"class_name": "FalsePositives", "config": {"name": "false_positives_10", "dtype": "float32", "thresholds": null}}, {"class_name": "FalseNegatives", "config": {"name": "false_negatives_10", "dtype": "float32", "thresholds": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "first_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 68]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 68]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "first_input"}}
õ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+µ&call_and_return_all_conditional_losses
¶__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 68}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68]}}
÷

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+·&call_and_return_all_conditional_losses
¸__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
÷

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
+¹&call_and_return_all_conditional_losses
º__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
÷

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
+»&call_and_return_all_conditional_losses
¼__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_73", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_73", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
÷

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
+½&call_and_return_all_conditional_losses
¾__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_74", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
÷

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+¿&call_and_return_all_conditional_losses
À__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_75", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_75", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
÷

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+Á&call_and_return_all_conditional_losses
Â__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_76", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_76", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
õ

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
+Ã&call_and_return_all_conditional_losses
Ä__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 43, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
§
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_rate
Emomentum_cachemmmmmm"m#m(m)m.m/m4m5m:m ;m¡v¢v£v¤v¥v¦v§"v¨#v©(vª)v«.v¬/v­4v®5v¯:v°;v±"
	optimizer

0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15"
trackable_list_wrapper

0
1
2
3
4
5
"6
#7
(8
)9
.10
/11
412
513
:14
;15"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
Flayer_metrics
Glayer_regularization_losses
	variables
trainable_variables
Hnon_trainable_variables

Ilayers
regularization_losses
Jmetrics
´__call__
³_default_save_signature
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
-
Åserving_default"
signature_map
": 	D2dense_70/kernel
:2dense_70/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Klayer_metrics
Llayer_regularization_losses
	variables
trainable_variables
Mnon_trainable_variables

Nlayers
regularization_losses
Ometrics
¶__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_71/kernel
:2dense_71/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Player_metrics
Qlayer_regularization_losses
	variables
trainable_variables
Rnon_trainable_variables

Slayers
regularization_losses
Tmetrics
¸__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_72/kernel
:2dense_72/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ulayer_metrics
Vlayer_regularization_losses
	variables
trainable_variables
Wnon_trainable_variables

Xlayers
 regularization_losses
Ymetrics
º__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_73/kernel
:2dense_73/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Zlayer_metrics
[layer_regularization_losses
$	variables
%trainable_variables
\non_trainable_variables

]layers
&regularization_losses
^metrics
¼__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_74/kernel
:2dense_74/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
_layer_metrics
`layer_regularization_losses
*	variables
+trainable_variables
anon_trainable_variables

blayers
,regularization_losses
cmetrics
¾__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_75/kernel
:2dense_75/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
dlayer_metrics
elayer_regularization_losses
0	variables
1trainable_variables
fnon_trainable_variables

glayers
2regularization_losses
hmetrics
À__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_76/kernel
:2dense_76/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
°
ilayer_metrics
jlayer_regularization_losses
6	variables
7trainable_variables
knon_trainable_variables

llayers
8regularization_losses
mmetrics
Â__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 :	+2output/kernel
:+2output/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
nlayer_metrics
olayer_regularization_losses
<	variables
=trainable_variables
pnon_trainable_variables

qlayers
>regularization_losses
rmetrics
Ä__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
J
s0
t1
u2
v3
w4
x5"
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	ytotal
	zcount
{	variables
|	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	}total
	~count

_fn_kwargs
	variables
	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}


thresholds
accumulator
	variables
	keras_api"¼
_tf_keras_metric¡{"class_name": "TruePositives", "name": "true_positives_10", "dtype": "float32", "config": {"name": "true_positives_10", "dtype": "float32", "thresholds": null}}


thresholds
accumulator
	variables
	keras_api"¼
_tf_keras_metric¡{"class_name": "TrueNegatives", "name": "true_negatives_10", "dtype": "float32", "config": {"name": "true_negatives_10", "dtype": "float32", "thresholds": null}}


thresholds
accumulator
	variables
	keras_api"¿
_tf_keras_metric¤{"class_name": "FalsePositives", "name": "false_positives_10", "dtype": "float32", "config": {"name": "false_positives_10", "dtype": "float32", "thresholds": null}}


thresholds
accumulator
	variables
	keras_api"¿
_tf_keras_metric¤{"class_name": "FalseNegatives", "name": "false_negatives_10", "dtype": "float32", "config": {"name": "false_negatives_10", "dtype": "float32", "thresholds": null}}
:  (2total
:  (2count
.
y0
z1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2accumulator
(
0"
trackable_list_wrapper
.
	variables"
_generic_user_object
(:&	D2Nadam/dense_70/kernel/m
": 2Nadam/dense_70/bias/m
):'
2Nadam/dense_71/kernel/m
": 2Nadam/dense_71/bias/m
):'
2Nadam/dense_72/kernel/m
": 2Nadam/dense_72/bias/m
):'
2Nadam/dense_73/kernel/m
": 2Nadam/dense_73/bias/m
):'
2Nadam/dense_74/kernel/m
": 2Nadam/dense_74/bias/m
):'
2Nadam/dense_75/kernel/m
": 2Nadam/dense_75/bias/m
):'
2Nadam/dense_76/kernel/m
": 2Nadam/dense_76/bias/m
&:$	+2Nadam/output/kernel/m
:+2Nadam/output/bias/m
(:&	D2Nadam/dense_70/kernel/v
": 2Nadam/dense_70/bias/v
):'
2Nadam/dense_71/kernel/v
": 2Nadam/dense_71/bias/v
):'
2Nadam/dense_72/kernel/v
": 2Nadam/dense_72/bias/v
):'
2Nadam/dense_73/kernel/v
": 2Nadam/dense_73/bias/v
):'
2Nadam/dense_74/kernel/v
": 2Nadam/dense_74/bias/v
):'
2Nadam/dense_75/kernel/v
": 2Nadam/dense_75/bias/v
):'
2Nadam/dense_76/kernel/v
": 2Nadam/dense_76/bias/v
&:$	+2Nadam/output/kernel/v
:+2Nadam/output/bias/v
Þ2Û
D__inference_model_10_layer_call_and_return_conditional_losses_688689
D__inference_model_10_layer_call_and_return_conditional_losses_688629
D__inference_model_10_layer_call_and_return_conditional_losses_688359
D__inference_model_10_layer_call_and_return_conditional_losses_688315À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
!__inference__wrapped_model_688094º
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
first_inputÿÿÿÿÿÿÿÿÿD
ò2ï
)__inference_model_10_layer_call_fn_688726
)__inference_model_10_layer_call_fn_688522
)__inference_model_10_layer_call_fn_688441
)__inference_model_10_layer_call_fn_688763À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_70_layer_call_and_return_conditional_losses_688774¢
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
Ó2Ð
)__inference_dense_70_layer_call_fn_688783¢
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
î2ë
D__inference_dense_71_layer_call_and_return_conditional_losses_688794¢
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
Ó2Ð
)__inference_dense_71_layer_call_fn_688803¢
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
î2ë
D__inference_dense_72_layer_call_and_return_conditional_losses_688814¢
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
Ó2Ð
)__inference_dense_72_layer_call_fn_688823¢
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
î2ë
D__inference_dense_73_layer_call_and_return_conditional_losses_688834¢
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
Ó2Ð
)__inference_dense_73_layer_call_fn_688843¢
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
î2ë
D__inference_dense_74_layer_call_and_return_conditional_losses_688854¢
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
Ó2Ð
)__inference_dense_74_layer_call_fn_688863¢
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
î2ë
D__inference_dense_75_layer_call_and_return_conditional_losses_688874¢
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
Ó2Ð
)__inference_dense_75_layer_call_fn_688883¢
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
î2ë
D__inference_dense_76_layer_call_and_return_conditional_losses_688894¢
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
Ó2Ð
)__inference_dense_76_layer_call_fn_688903¢
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
ì2é
B__inference_output_layer_call_and_return_conditional_losses_688914¢
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
Ñ2Î
'__inference_output_layer_call_fn_688923¢
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
ÏBÌ
$__inference_signature_wrapper_688569first_input"
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
 
!__inference__wrapped_model_688094y"#()./45:;4¢1
*¢'
%"
first_inputÿÿÿÿÿÿÿÿÿD
ª "/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ+¥
D__inference_dense_70_layer_call_and_return_conditional_losses_688774]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿD
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_70_layer_call_fn_688783P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿD
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_71_layer_call_and_return_conditional_losses_688794^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_71_layer_call_fn_688803Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_72_layer_call_and_return_conditional_losses_688814^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_72_layer_call_fn_688823Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_73_layer_call_and_return_conditional_losses_688834^"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_73_layer_call_fn_688843Q"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_74_layer_call_and_return_conditional_losses_688854^()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_74_layer_call_fn_688863Q()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_75_layer_call_and_return_conditional_losses_688874^./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_75_layer_call_fn_688883Q./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_76_layer_call_and_return_conditional_losses_688894^450¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_76_layer_call_fn_688903Q450¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¿
D__inference_model_10_layer_call_and_return_conditional_losses_688315w"#()./45:;<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿD
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 ¿
D__inference_model_10_layer_call_and_return_conditional_losses_688359w"#()./45:;<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿD
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 º
D__inference_model_10_layer_call_and_return_conditional_losses_688629r"#()./45:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿD
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 º
D__inference_model_10_layer_call_and_return_conditional_losses_688689r"#()./45:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿD
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 
)__inference_model_10_layer_call_fn_688441j"#()./45:;<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿD
p

 
ª "ÿÿÿÿÿÿÿÿÿ+
)__inference_model_10_layer_call_fn_688522j"#()./45:;<¢9
2¢/
%"
first_inputÿÿÿÿÿÿÿÿÿD
p 

 
ª "ÿÿÿÿÿÿÿÿÿ+
)__inference_model_10_layer_call_fn_688726e"#()./45:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿD
p

 
ª "ÿÿÿÿÿÿÿÿÿ+
)__inference_model_10_layer_call_fn_688763e"#()./45:;7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿD
p 

 
ª "ÿÿÿÿÿÿÿÿÿ+£
B__inference_output_layer_call_and_return_conditional_losses_688914]:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ+
 {
'__inference_output_layer_call_fn_688923P:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ+±
$__inference_signature_wrapper_688569"#()./45:;C¢@
¢ 
9ª6
4
first_input%"
first_inputÿÿÿÿÿÿÿÿÿD"/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ+