from flask import Flask, render_template, url_for, request, make_response ,redirect, jsonify,request,flash
from simdyn import main_sim
from websim import load_sim,LoadFile
from werkzeug.utils import secure_filename
import module_shared as sh
import os, time, glob, datetime
from flask_executor import Executor
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager,login_user,current_user,logout_user,login_required,UserMixin
from flask_mail import Mail,Message
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer 
from flask_wtf import FlaskForm
from flask_login import current_user
from wtforms import StringField, PasswordField,SubmitField,BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo,ValidationError


application = app = Flask(__name__)
executor = Executor(app)

app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 2
app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True
app.config['SECRET_KEY'] ='eee17b913b3123f3f64f4e164e4925c1' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASSWORD')

time_stamp = None
filenames = None
simtype = None 
sh.log_data = "Starting Simulation"
class RegisterForm(FlaskForm):
	username = StringField("Username",
							validators = [DataRequired(),Length(min = 2,max = 20)])
	email = StringField('Email',validators = [DataRequired(),Email()])
	password = PasswordField("Password",validators= [DataRequired(),Length(min = 8, max = 12)])
	confirm_password = PasswordField("Confirm Password",
				  		validators= [DataRequired(),Length(min = 8,max = 12),EqualTo('password')])
	submit = SubmitField("Register")

	def validate_username(self,username):
		user = User.query.filter_by(username = username.data).first()
		if user:
			raise ValidationError('This username already exists. Please choose a different username')
	def validate_email(self,email):
		user = User.query.filter_by(email = email.data).first()
		if user:
			raise ValidationError('This email already exists. Please choose a different email')

class LoginForm(FlaskForm):
	email = StringField('Email',validators = [DataRequired(),Email()])
	password = PasswordField("Password",validators= [DataRequired(),Length(min = 8, max = 12)])
	remember = BooleanField("Remember me")
	submit = SubmitField("Login")

class UpdateAccountForm(FlaskForm):
	username = StringField("Username",
							validators = [DataRequired(),Length(min = 2,max = 20)])
	email = StringField('Email',validators = [DataRequired(),Email()])
	submit = SubmitField("Update")

	def validate_username(self,username):
		if username.data != current_user.username:
			user = User.query.filter_by(username = username.data).first()
			if user:
				raise ValidationError('This username already exists. Please choose a different username')
	def validate_email(self,email):
		if email.data != current_user.email:
			user = User.query.filter_by(email = email.data).first()
			if user:
				raise ValidationError('This email already exists. Please choose a different email')

class RequestResetForm(FlaskForm):
	email = StringField('Email',validators = [DataRequired(),Email()])
	submit = SubmitField("Request Password Reset")
	def validate_email(self,email):
		user = User.query.filter_by(email = email.data).first()
		if user is None:
			raise ValidationError('There is no account with this email.Please choose a valid email')

class ResetPasswordForm(FlaskForm):
	password = PasswordField('Password',validators= [DataRequired(),Length(min = 8, max = 12)])
	confirm_password = PasswordField("Confirm Password",
				  		validators= [DataRequired(),Length(min = 8),EqualTo('password')])
	submit = SubmitField('Reset Password')

@login_manager.user_loader
def load_user(user_id):
	return User.query.get(int(user_id))

class User(db.Model,UserMixin):
	id = db.Column(db.Integer,primary_key = True)
	username = db.Column(db.String(20),nullable = False)
	email = db.Column(db.String(120),unique = True,nullable = False)
	password = db.Column(db.String(60),nullable = False)
	
	def get_reset_token(self,expires_sec = 1800):
		s = Serializer(app.config['SECRET_KEY'],expires_sec)
		return s.dumps({'user_id':self.id}).decode('utf-8')

	@staticmethod
	def verify_reset_token(token):
		s = Serializer(app.config['SECRET_KEY'])
		try:
			user_id = s.loads(token)['user_id']
		except:
			return None
		return User.query.get(user_id)

	def __repr__(self):
		return f"User('{self.username}','{self.email}')"


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html',title='Home')

@app.route("/simulation")
def simulation():
    return render_template('simulation.html',title='Simulation')

@app.route("/instruction")
def instruction():
    return render_template('instruction.html',title='Instruction')

@app.route("/contact")
def contact():
    return render_template('contact.html',title='Contact')

@app.route("/theory")
def theory():
    return render_template('theory.html',title='Theory')

@app.route("/wavesim")
def wavesim():
    return render_template('visualization.html',title='WaveSim',meshdat = LoadFile("KCS.GDF",False),state = False)

@app.route("/visual")
def visual():
	return render_template('visualization.html',title='Visual', state = True)

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/input_action",methods=['POST','GET'])
def input():
	global time_stamp,filenames,simtype
	simtype = True
	if not os.path.isdir('input_temp'):
		os.mkdir('input_temp')
	else:
		for filename in glob.glob(os.path.join('input_temp', '*.*')):
			os.remove(filename)

	if request.method == 'POST':
		time_stamp = str(time.time())
		gdfname = []
		count = 1
		files = request.files.getlist('gdffiles')
		for file in files:
			filename = secure_filename(file.filename)
			gdfname += ['input_temp/'+ time_stamp + filename + str(count)]
			file.save(gdfname[-1])
			count +=1

		files = request.files.getlist('FDfile')
		for file in files:
			filename = secure_filename(file.filename)
			fdfname = 'input_temp/'+ time_stamp + filename
			file.save(fdfname)

		files = request.files.getlist('AMDfile')
		for file in files:
			filename = secure_filename(file.filename)
			amdfname = 'input_temp/'+ time_stamp + filename
			file.save(amdfname)

		files = request.files.getlist('inpfile')
		for file in files:
			filename = secure_filename(file.filename)
			inpname = 'input_temp/'+ time_stamp + filename
			file.save(inpname)

		filenames = [gdfname,fdfname,amdfname,inpname]

		executor.submit_stored(time_stamp,main_sim,time_stamp,filenames)

	return redirect(url_for('loadingpage'))

@app.route("/cancel_simulation")
def cancel_simulation():
	global time_stamp
	if (executor.futures.cancel(time_stamp)):
		return redirect(url_for('simulation'))
	else:
		return render_template('loading.html', title='simulation',status = "Cancel Attempt failed")

@app.route("/load_action",methods=['POST','GET'])
def loadinput():

	global time_stamp,filenames,simtype
	simtype = False
	if not os.path.isdir('input_temp'):
		os.mkdir('input_temp')
	else:
		for filename in glob.glob(os.path.join('input_temp', '*.*')):
			os.remove(filename)
	time_stamp = str(time.time())

	if request.method == 'POST':
		time_stamp = str(time.time())
		files = request.files.getlist('loadfiles')
		for file in files:
			filename = secure_filename(file.filename)
			loadfname = 'input_temp/'+ time_stamp + filename
			file.save(loadfname)
		filenames = loadfname

	executor.submit_stored(time_stamp,load_sim,filenames)
	return redirect(url_for('loadingpage'))

@app.route("/loadingpage")
def loadingpage():
	if not executor.futures.done(time_stamp):
		return render_template('loading.html', title='simulation',status = "Running")
	return redirect(url_for('result'))


@app.route('/update_log', methods=['POST'])
def update_log():
	return jsonify({
        'value': sh.log_data,
		'frame': sh.frame_log
	})

@app.route("/result")
def result():
	global time_stamp,filenames,simtype
	if not executor.futures.done(time_stamp):
		return redirect(url_for('loadingpage'))
	future = executor.futures.pop(time_stamp)
	future.result()
	return render_template('result.html', title='simulation',plotnames = sh.plot_filenames, ssname = sh.ssname, simname = sh.simfilename)

@app.route("/register",methods = ['GET','POST'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegisterForm()
	if form.validate_on_submit():
		hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user = User(username = form.username.data, email = form.email.data, password = hashed_pw)
		db.session.add(user)
		db.session.commit()
		flash('Your account has been created successfully! You can login now','success')
		return redirect(url_for('login'))
	return render_template('register.html',title = "Register",form = form)
	

@app.route("/login",methods = ['GET','POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		if user and bcrypt.check_password_hash(user.password,form.password.data):
			login_user(user,remember = form.remember.data)
			next_page = request.args.get('next')	
			flash('Login Successful!','success')
			return redirect(next_page) if next_page else redirect(url_for('home'))
		else:
			flash('Login Unsuccessful. Please check email and password','danger')
	return render_template('login.html',title = "Login",form = form)

@app.route("/logout")
def logout():
	logout_user()
	return redirect(url_for('home'))

@app.route("/account",methods = ['GET','POST'])
@login_required
def account():
	form = UpdateAccountForm()
	if form.validate_on_submit():
		current_user.username = form.username.data
		current_user.email = form.email.data
		db.session.commit()
		flash('Your account has been updated!','success')
		return redirect(url_for('account'))
	elif request.method == 'GET':
		form.username.data = current_user.username
		form.email.data = current_user.email
	return render_template('account.html',title = "Account",form = form)

def send_reset_email(user):
	token = user.get_reset_token()
	msg = Message('Password Reset Request', sender = 'noreply@demo.com',
				recipients = [user.email])
	msg.body = ''' To reset your password , visit the following link:
	{url_for('reset_token',token = token,_external = True)}

	If you did not make this request them simply ignore this mail.
	'''

@app.route("/reset_password",methods = ['GET','POST'])
def reset_request():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RequestResetForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email = form.email.data).first()
		send_reset_email(user)
		flash('An email has been sent with instructions to reset your password')
		return redirect(url_for('login'))
	return render_template('reset_request.html',title = 'Reset Password',form = form)

@app.route("/reset_password/<token>",methods = ['GET','POST'])
def reset_token(token):
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	user = User.verify_reset_token(token)
	if user is None:
		flash('That is an invalid token','warning')
		return redirect(url_for('reset_request'))
	form = ResetPasswordForm()
	if form.validate_on_submit():
		hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user.password = hashed_pw
		db.session.commit()
		flash('Your password has been updated! You can login now','success')
		return redirect(url_for('login'))
	return render_template('reset_token.html',title = 'Reset Password',form = form)

if __name__ == '__main__':
    app.run(debug=True)
