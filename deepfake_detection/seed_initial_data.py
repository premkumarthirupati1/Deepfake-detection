"""
Database seed script to populate initial users, records, and accuracies.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detection.settings')
django.setup()

from Remote_User.models import ClientRegister_Model, deepfake_detection_type, detection_accuracy, detection_ratio

def seed_database():
    print("Seeding database...")

    # Seed Users
    users_data = [
        {
            'username': 'Rajesh',
            'email': 'Rajesh123@gmail.com',
            'password': 'Rajesh',
            'phoneno': '9535866270',
            'country': 'India',
            'state': 'Karnataka',
            'city': 'Bangalore'
        },
        {
            'username': 'Manjunath',
            'email': 'tmksmanju13@gmail.com',
            'password': 'Manjunath',
            'phoneno': '9535866270',
            'country': 'India',
            'state': 'Karnataka',
            'city': 'Bangalore'
        }
    ]

    for u in users_data:
        ClientRegister_Model.objects.get_or_create(
            username=u['username'],
            defaults=u
        )

    # Seed Predictions
    predictions_data = [
        {
            'url': 'http://www.avevaconstruction.com/blesstool/image.htm',
            'ipaddress': '175.180.184.106',
            'videoname': 'lxnqzocgaq.mp4',
            'original_width': '223',
            'original_height': '217',
            'country': 'Taiwan',
            'locale': 'Taipei',
            'latitude': '25.0392',
            'longitude': '121.525',
            'Prediction': 'FAKE'
        },
        {
            'url': 'https://www.velocidrone.com/',
            'ipaddress': '213.215.43.23',
            'videoname': 'ddvgrczjno.mp4',
            'original_width': '155',
            'original_height': '155',
            'country': 'France',
            'locale': 'NA',
            'latitude': '48.86',
            'longitude': '2.35',
            'Prediction': 'REAL'
        },
        {
            'url': 'http://www.tutorialspoint.com/dbms/',
            'ipaddress': '219.148.38.70',
            'videoname': 'sjgbiywngj.mp4',
            'original_width': '185',
            'original_height': '186',
            'country': 'China',
            'locale': 'Hebei',
            'latitude': '39.8897',
            'longitude': '115.275',
            'Prediction': 'FAKE'
        },
        {
            'url': 'http://support-appleld.com.secureupdate.duilawyeryork.com/ap/4a48dae8675698c',
            'ipaddress': '198.20.69.74',
            'videoname': 'ypewpvpfmk.mp4',
            'original_width': '186',
            'original_height': '186',
            'country': 'United States',
            'locale': 'Illinois',
            'latitude': '41.8825',
            'longitude': '-87.6441',
            'Prediction': 'REAL'
        },
        {
            'url': 'http://creative-punch.net/2014/10/intro-html5-speech-synthesis-api/',
            'ipaddress': '202.85.214.186',
            'videoname': 'nuzbbjlaed.mp4',
            'original_width': '154',
            'original_height': '155',
            'country': 'China',
            'locale': 'NA',
            'latitude': '35',
            'longitude': '105',
            'Prediction': 'FAKE'
        }
    ]

    for p in predictions_data:
        deepfake_detection_type.objects.get_or_create(
            videoname=p['videoname'],
            defaults=p
        )

    # Seed Accuracies
    accuracies = [
        ('Naive Bayes', '56.92'),
        ('SVM', '56.85'),
        ('Logistic Regression', '59.89'),
        ('Decision Tree Classifier', '58.23'),
        ('GradientBoostingClassifier', '64.27'),
        ('Voting Ensemble', '61.45'),
    ]
    detection_accuracy.objects.all().delete()
    for name, ratio in accuracies:
        detection_accuracy.objects.create(names=name, ratio=ratio)

    # Seed Ratios
    detection_ratio.objects.all().delete()
    detection_ratio.objects.create(names='FAKE', ratio='60.0')
    detection_ratio.objects.create(names='REAL', ratio='40.0')

    print("Database seeding completed successfully!")

if __name__ == '__main__':
    seed_database()
