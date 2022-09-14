### Locally accessible edge computers
To run ansible playbooks like you would from a notebook. this speeds it up a bit

1. Add your public ssh key to .ssh/authorized_keys for the ansible_user you will use, **on the edge computer**
2. Update hacks/inventory with host/ansible_host/ansible_user... to match your machine(s)

Run:
```bash
ANSIBLE_CONFIG=hacks/ansible.cfg ansible-playbook -v -i hacks/inventory edgebuild.yaml --limit <device>
```


