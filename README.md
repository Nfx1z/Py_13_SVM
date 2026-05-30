### UDP Silat

Reminder:
```bash
# Windows (run as admin)
netsh advfirewall firewall add rule name="Silat UDP" protocol=UDP dir=in localport=5005 action=allow
```

```bash
# Linux
sudo ufw allow 5005/udp
```