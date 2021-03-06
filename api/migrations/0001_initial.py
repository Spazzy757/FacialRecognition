# Generated by Django 2.1 on 2018-08-28 05:45

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
                ('labels', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=256), size=None)),
                ('generated_labels', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=256), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.ImageField(upload_to='')),
                ('remark', models.CharField(max_length=20)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Match',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('dog', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.Dog')),
                ('file', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.File')),
            ],
        ),
        migrations.AddField(
            model_name='dog',
            name='picture',
            field=models.ManyToManyField(blank=True, through='api.Match', to='api.File'),
        ),
    ]
