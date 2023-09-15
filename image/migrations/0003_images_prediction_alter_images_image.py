# Generated by Django 4.0.2 on 2022-06-19 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image', '0002_alter_images_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='images',
            name='prediction',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='images',
            name='image',
            field=models.FileField(default='', upload_to='images/imagedata'),
        ),
    ]
