# Generated by Django 4.0.2 on 2022-06-21 07:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image', '0003_images_prediction_alter_images_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='images',
            name='choice',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='images',
            name='image2',
            field=models.FileField(default='', upload_to='images/imagedata'),
        ),
    ]