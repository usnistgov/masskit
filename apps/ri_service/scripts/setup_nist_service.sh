#!/bin/sh
# Please run this script as sudo
cp nist_service.service /etc/systemd/system/
systemctl start nist_service
systemctl enable nist_service

cp nist_service /etc/nginx/sites-available/
ln -s /etc/nginx/sites-available/nist_service /etc/nginx/sites-enabled
service nginx configtest
service nginx restart
# nginx log file is in /var/log/nginx
