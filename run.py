

from skindisease import app, db

# Create tables if not already created
with app.app_context():
    db.create_all()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


