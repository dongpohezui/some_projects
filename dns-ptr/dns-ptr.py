
from dns import reversename, resolver
import IPy
import argparse

def getptr(ip):
	domain_address = reversename.from_address(ip)
	try:
		domain_name = resolver.resolve(domain_address, 'PTR')
		answer = domain_name.response.answer[0]
	except :
		answer = []
	return answer

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Test for argparse')
	parser.add_argument('ips', metavar='N', type=str, nargs='+',
                    help='ips')
	args = parser.parse_args()
	print(args.ips )
	for arg in args.ips :
		ips=IPy.IP(arg)
		for ip in ips:
			ip = str(ip)
			answer = getptr(ip)
			for i in range(len(answer)):
				print(ip,str(answer[i])[:-1])
		
